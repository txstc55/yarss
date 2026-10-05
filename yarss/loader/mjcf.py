"""Use MuJoCo's compiler to resolve MJCF defaults, includes, and orientations."""

from pathlib import Path

import numpy as np

from ..joints import (
  BallJoint,
  FixedJoint,
  FloatingJoint,
  JointLimit,
  RotatingJoint,
  SlidingJoint,
  UnlimitedRotatingJoint,
)
from ..parts import Geometry, Part
from ..robot import Robot
from ..transforms import transform


def _generated_name(prefix: str, index: int, used: set[str]) -> str:
  """Generated names must not shadow a name authored in the file."""
  name = f"__{prefix}_{index}"
  while name in used:
    name += "_"
  used.add(name)
  return name


def load_mjcf(path: Path) -> Robot:
  import mujoco

  # Recent MuJoCo versions dispatch by extension and do not recognize .mjcf.
  # A virtual .xml name keeps includes/assets relative to the original folder.
  if path.suffix.lower() == ".xml":
    spec = mujoco.MjSpec.from_file(str(path))
  else:
    xml_name = path.name + ".xml"
    spec = mujoco.MjSpec.from_file(
      str(path.parent / xml_name), include={xml_name: path.read_bytes()}
    )
  # Retain source links and visual geometry even if the file requests fusion.
  spec.compiler.fusestatic = False
  spec.compiler.discardvisual = False
  spec.compiler.alignfree = False
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)  # Default pose only; no simulation step.
  robot = Robot(spec.modelname or path.stem, path, metadata={"format": "mjcf", "up_axis": "Z"})
  names = {0: None}
  body_names = {model.body(i).name for i in range(1, model.nbody)}
  joint_names = {model.joint(i).name for i in range(model.njnt)}
  for body_id in range(1, model.nbody):
    name = model.body(body_id).name or _generated_name("body", body_id, body_names)
    names[body_id] = name
    robot.add_part(
      Part(
        name,
        transform=transform(data.xpos[body_id], data.xquat[body_id]),
        local_transform=transform(model.body_pos[body_id], model.body_quat[body_id]),
        metadata={"mass": float(model.body_mass[body_id])},
      )
    )

  world_geometry = []
  # Explicit contact pairs can enable collision even when both masks are zero.
  paired_geoms = set(model.pair_geom1) | set(model.pair_geom2)
  for geom_id in range(model.ngeom):
    shape = _geometry(model, geom_id, mujoco)
    body_id = int(model.geom_bodyid[geom_id])
    if body_id == 0:
      world_geometry.append(shape)
      continue
    part = robot.parts[names[body_id]]
    # MuJoCo shows groups 0, 1, and 2 by default. FR3 puts its separate
    # collision meshes in group 3; drawing both produces overlapping surfaces.
    if 0 <= model.geom_group[geom_id] <= 2:
      part.visuals.append(shape)
    if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id] or geom_id in paired_geoms:
      part.collisions.append(shape)
  robot.metadata["world_geometry"] = world_geometry

  joint_types = {
    mujoco.mjtJoint.mjJNT_FREE: FloatingJoint,
    mujoco.mjtJoint.mjJNT_BALL: BallJoint,
    mujoco.mjtJoint.mjJNT_SLIDE: SlidingJoint,
    mujoco.mjtJoint.mjJNT_HINGE: RotatingJoint,
  }
  for body_id in range(1, model.nbody):
    parent = robot.parts.get(names[int(model.body_parentid[body_id])])
    child = robot.parts[names[body_id]]
    body_pose = transform(model.body_pos[body_id], model.body_quat[body_id])
    start, count = int(model.body_jntadr[body_id]), int(model.body_jntnum[body_id])
    if count == 0:
      robot.add_joint(
        FixedJoint(
          _generated_name("fixed", body_id, joint_names),
          parent,
          child,
          parent_frame=body_pose,
          metadata={"implicit": True},
        )
      )
    for joint_id in range(start, start + count):
      kind = joint_types[int(model.jnt_type[joint_id])]
      limited = bool(model.jnt_limited[joint_id])
      if kind is RotatingJoint and not limited:
        kind = UnlimitedRotatingJoint
      stiffness = float(model.jnt_stiffness[joint_id])
      if kind in {RotatingJoint, UnlimitedRotatingJoint, BallJoint}:
        # Torque per degree, so stiffness * a degree displacement is still N*m.
        stiffness *= np.pi / 180
      anchor = transform(model.jnt_pos[joint_id])
      joint = kind(
        model.joint(joint_id).name or _generated_name("joint", joint_id, joint_names),
        parent,
        child,
        parent_frame=body_pose @ anchor,
        child_frame=anchor,
        axis=model.jnt_axis[joint_id].copy(),
        metadata={
          "order_in_body": joint_id - start,
          "stiffness": stiffness,
        },
      )
      address = int(model.jnt_qposadr[joint_id])
      if kind is FloatingJoint:
        joint.initial_value = transform(
          model.qpos0[address : address + 3], model.qpos0[address + 3 : address + 7]
        )
      elif kind is BallJoint:
        joint.initial_value = transform(quaternion=model.qpos0[address : address + 4])[
          :3, :3
        ].copy()
      elif kind is SlidingJoint:
        joint.initial_value = float(model.qpos0[address])
      else:
        # MuJoCo compiles angular coordinates to radians, regardless of XML units.
        joint.initial_value = float(np.rad2deg(model.qpos0[address]))
      # Matrices need separate storage so editing the current value preserves the loaded value.
      joint.value = (
        joint.initial_value.copy()
        if isinstance(joint.initial_value, np.ndarray)
        else joint.initial_value
      )
      if limited:
        bounds = model.jnt_range[joint_id]
        if kind is not SlidingJoint:
          bounds = np.rad2deg(bounds)
        lower, upper = (float(v) for v in bounds)
        key = "angle" if kind is BallJoint else "value"
        joint.limits[key] = JointLimit(lower, upper)
      robot.add_joint(joint)
  # Actuators, tendons, and equality constraints are outside this geometry/
  # joint-definition layer. Record their presence rather than simulating them.
  robot.metadata.update(
    actuator_count=model.nu, tendon_count=model.ntendon, equality_constraint_count=model.neq
  )
  return robot


def _geometry(model, geom_id: int, mujoco) -> Geometry:
  kind = mujoco.mjtGeom(int(model.geom_type[geom_id])).name.removeprefix("mjGEOM_").lower()
  shape = Geometry(
    kind,
    transform=transform(model.geom_pos[geom_id], model.geom_quat[geom_id]),
    source=model.geom(geom_id).name or f"geom_{geom_id}",
  )
  x, y, z = (float(v) for v in model.geom_size[geom_id])
  if kind == "mesh":
    mesh_id = int(model.geom_dataid[geom_id])
    first = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    shape.vertices = model.mesh_vert[first : first + count].astype(np.float64, copy=True)
    first = int(model.mesh_faceadr[mesh_id])
    count = int(model.mesh_facenum[mesh_id])
    shape.faces = tuple(
      tuple(int(v) for v in face) for face in model.mesh_face[first : first + count]
    )
    # Compiled vertices and geom poses already include mesh scale and the
    # compiler's recentering transform; applying the source scale again is wrong.
  elif kind in {"box", "plane"}:
    shape.parameters = {"size": (2 * x, 2 * y, 2 * z)}
  elif kind in {"capsule", "cylinder"}:
    shape.parameters = {"radius": x, "length": 2 * y}
  elif kind == "sphere":
    shape.parameters = {"radius": x}
  elif kind == "ellipsoid":
    shape.parameters = {"radii": (x, y, z)}
  else:
    raise ValueError(f"Unsupported MJCF geometry type {kind!r}; cannot extract its shape")
  return shape
