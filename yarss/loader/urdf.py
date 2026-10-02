"""Read URDF links, geometry, and joint definitions using the standard library."""

from collections import deque
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

import numpy as np

from ..joints import (
  FixedJoint,
  FloatingJoint,
  JointLimit,
  Mimic,
  PlanarJoint,
  RotatingJoint,
  SlidingJoint,
  UnlimitedRotatingJoint,
)
from ..parts import Geometry, Part
from ..robot import Robot
from ..transforms import Vector3, rpy_quaternion, transform

JOINT_TYPES = {
  "fixed": FixedJoint,
  "revolute": RotatingJoint,
  "continuous": UnlimitedRotatingJoint,
  "prismatic": SlidingJoint,
  "planar": PlanarJoint,
  "floating": FloatingJoint,
}


def vector(text: str, count: int = 3) -> Vector3:
  values = np.array([float(v) for v in text.split()])
  if len(values) != count or not np.isfinite(values).all():
    raise ValueError(f"Expected {count} finite numbers, found {text!r}")
  return values


def origin(element: ET.Element):
  node = element.find("origin")
  if node is None:
    return np.eye(4)
  return transform(
    vector(node.get("xyz", "0 0 0")), rpy_quaternion(vector(node.get("rpy", "0 0 0")))
  )


def mesh_path(uri: str, source: Path, packages: dict[str, Path]) -> Path:
  parsed = urlparse(uri)
  if parsed.scheme == "package":
    package = parsed.netloc
    relative = unquote(parsed.path.lstrip("/"))
    if package in packages:
      path = packages[package] / relative
    else:
      # A vendored ROS package can be loaded without installing ROS.
      candidates = [ancestor / relative for ancestor in source.parents if ancestor.name == package]
      path = next((p for p in candidates if p.is_file()), None)
      if path is None:
        raise FileNotFoundError(
          f"Cannot resolve {uri!r}; pass package_paths={{'{package}': '/path/to/package'}}"
        )
  elif parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
    path = Path(unquote(parsed.path))
  elif not parsed.scheme:
    path = source.parent / unquote(uri)
  else:
    raise ValueError(f"Unsupported mesh URI: {uri!r}")
  path = path.resolve()
  if not path.is_file():
    raise FileNotFoundError(f"Mesh {uri!r} referenced by {source}: {path}")
  return path


def geometry(element: ET.Element, source: Path, packages: dict[str, Path]) -> Geometry:
  shape = element.find("geometry")
  if shape is None or len(shape) != 1:
    raise ValueError("A URDF visual/collision must contain exactly one geometry")
  node = shape[0]
  result = Geometry(kind=node.tag, transform=origin(element))
  if node.tag == "mesh":
    result.source = node.attrib["filename"]
    result.mesh_path = mesh_path(result.source, source, packages)
    result.vertices, result.faces = _read_mesh(result.mesh_path)
    result.transform = result.transform @ transform(scale=vector(node.get("scale", "1 1 1")))
  elif node.tag == "box":
    result.parameters = {"size": vector(node.attrib["size"])}
  elif node.tag == "cylinder":
    result.parameters = {
      "radius": float(node.attrib["radius"]),
      "length": float(node.attrib["length"]),
    }
  elif node.tag == "sphere":
    result.parameters = {"radius": float(node.attrib["radius"])}
  else:
    raise ValueError(f"Unsupported URDF geometry: {node.tag!r}")
  return result


def _read_mesh(path: Path):
  """Read mesh geometry with scene transforms, without requiring textures."""
  if path.suffix.lower() == ".dae":
    import collada

    document = collada.Collada(str(path))
    if document.scene is None:
      raise ValueError(f"COLLADA file has no active scene: {path}")
    vertices, faces = [], []
    meters = document.assetInfo.unitmeter or 1.0
    for instance in document.scene.objects("geometry"):
      for primitive in instance.primitives():
        if hasattr(primitive, "triangleset"):
          primitive = primitive.triangleset()
        if not isinstance(primitive, collada.triangleset.BoundTriangleSet):
          raise ValueError(f"Unsupported non-surface COLLADA primitive in {path}")
        offset = len(vertices)
        vertices.extend(np.asarray(primitive.vertex, dtype=float) * meters)
        faces.extend(tuple(int(i) + offset for i in f) for f in primitive.vertex_index)
  else:
    import trimesh

    mesh = trimesh.load_mesh(path, process=False, skip_materials=True)
    if mesh.units:
      mesh.convert_units("meters")
    vertices = np.array(mesh.vertices, dtype=float, copy=True)
    faces = [tuple(int(i) for i in f) for f in mesh.faces]
  if len(vertices) == 0 or not faces:
    raise ValueError(f"No triangle mesh found in {path}")
  return np.asarray(vertices, dtype=float), tuple(faces)


def load_urdf(path: Path, root: ET.Element, packages: dict[str, Path]) -> Robot:
  robot = Robot(root.get("name", path.stem), path, metadata={"format": "urdf", "up_axis": "Z"})
  for link in root.findall("link"):
    part = Part(link.attrib["name"])
    part.visuals = [geometry(v, path, packages) for v in link.findall("visual")]
    part.collisions = [geometry(c, path, packages) for c in link.findall("collision")]
    inertial = link.find("inertial")
    if inertial is not None:
      part.metadata["inertial"] = ET.tostring(inertial, encoding="unicode")
    robot.add_part(part)

  for node in root.findall("joint"):
    kind = node.attrib["type"]
    if kind not in JOINT_TYPES:
      raise ValueError(f"Unsupported URDF joint type: {kind!r}")
    parent, child = node.find("parent"), node.find("child")
    if parent is None or child is None:
      raise ValueError(f"Joint {node.get('name')!r} requires parent and child links")
    joint = JOINT_TYPES[kind](
      node.attrib["name"],
      parent.attrib["link"],
      child.attrib["link"],
      parent_frame=origin(node),
    )
    axis = node.find("axis")
    values = vector(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0")
    length = np.linalg.norm(values)
    if length == 0:
      raise ValueError(f"Joint {joint.name!r} has a zero-length axis")
    joint.axis = values / length
    limit = node.find("limit")
    if limit is not None:
      limits = {
        key: float(limit.attrib[key])
        for key in ("lower", "upper", "effort", "velocity")
        if key in limit.attrib
      }
      if kind == "continuous":
        limits.pop("lower", None)
        limits.pop("upper", None)
      joint.limits["position"] = JointLimit(**limits)
    mimic = node.find("mimic")
    if mimic is not None:
      joint.mimic = Mimic(
        mimic.attrib["joint"],
        float(mimic.get("multiplier", "1")),
        float(mimic.get("offset", "0")),
      )
    dynamics = node.find("dynamics")
    if dynamics is not None:
      joint.metadata["dynamics"] = {k: float(v) for k, v in dynamics.attrib.items()}
    robot.add_joint(joint)

  robot.validate()
  _set_default_poses(robot)
  return robot


def _set_default_poses(robot: Robot) -> None:
  """URDF must be one connected tree; accumulate its zero-position origins."""
  outgoing = {name: [] for name in robot.parts}
  children = set()
  for joint in robot.joints.values():
    if joint.child in children:
      raise ValueError(f"URDF link {joint.child!r} has more than one parent")
    children.add(joint.child)
    outgoing[joint.parent].append(joint)
  roots = [name for name in robot.parts if name not in children]
  if len(roots) != 1:
    raise ValueError(f"URDF requires one root link; found {roots}")
  pending = deque(roots)
  visited = set()
  while pending:
    name = pending.popleft()
    if name in visited:
      raise ValueError("URDF contains a joint cycle")
    visited.add(name)
    for joint in outgoing[name]:
      robot.parts[joint.child].transform = robot.parts[name].transform @ joint.parent_frame
      pending.append(joint.child)
  if len(visited) != len(robot.parts):
    raise ValueError("URDF contains a disconnected joint cycle")
