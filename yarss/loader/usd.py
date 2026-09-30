"""Read composed OpenUSD geometry and standard UsdPhysics joint schemas."""

from math import isfinite, radians
from pathlib import Path

from ..joints import (
  ContinuousJoint,
  D6Joint,
  DistanceJoint,
  FixedJoint,
  JointLimit,
  PrismaticJoint,
  RevoluteJoint,
  SphericalJoint,
)
from ..parts import Geometry, Part
from ..robot import Robot
from ..transforms import multiply, transform


def _matrix(matrix, meters: float):
  # Gf matrices use row vectors; YARSS uses column vectors. Only translation
  # needs unit conversion here; vertices/dimensions are converted separately.
  return tuple(
    tuple(float(matrix[j][i]) * (meters if j == 3 and i < 3 else 1) for j in range(4))
    for i in range(4)
  )


def _quaternion(value):
  return (float(value.GetReal()), *(float(v) for v in value.GetImaginary()))


def _bound(value, factor: float):
  return float(value) * factor if value is not None and isfinite(value) else None


def load_usd(path: Path, root_path: str | None) -> Robot:
  try:
    from pxr import Usd, UsdGeom, UsdPhysics
  except ImportError as error:
    raise ImportError('USD loading requires: pip install "yarss[usd]"') from error

  stage = Usd.Stage.Open(str(path))
  if stage is None:
    raise ValueError(f"Cannot open USD stage: {path}")
  # Missing references otherwise become warnings and incomplete robots.
  if stage.GetCompositionErrors():
    raise ValueError(f"USD composition failed: {stage.GetCompositionErrors()}")
  root = stage.GetPrimAtPath(root_path) if root_path else stage.GetPseudoRoot()
  if not root:
    raise ValueError(f"USD root {root_path!r} does not exist in {path}")
  prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
  meters = float(UsdGeom.GetStageMetersPerUnit(stage))
  if not isfinite(meters) or meters <= 0:
    raise ValueError("USD metersPerUnit must be positive and finite")
  default_prim = stage.GetDefaultPrim()
  name = root.GetName() if root_path else default_prim.GetName() if default_prim else path.stem
  robot = Robot(
    str(name),
    path,
    metadata={
      "format": "usd",
      "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
      "source_meters_per_unit": meters,
      "articulation_roots": [
        str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)
      ],
    },
  )
  cache = UsdGeom.XformCache(Usd.TimeCode.Default())
  body_prims = {str(p.GetPath()): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
  joint_prims = [p for p in prims if p.IsA(UsdPhysics.Joint)]

  def endpoint(relation):
    targets = relation.GetTargets()
    if not targets:
      return None, None
    if len(targets) != 1:
      raise ValueError(f"Expected one body target on {relation.GetPath()}")
    target = stage.GetPrimAtPath(targets[0])
    if not target:
      raise ValueError(f"Joint references missing USD prim {targets[0]}")
    if not target.GetPath().HasPrefix(root.GetPath()):
      raise ValueError(f"Joint target {targets[0]} is outside selected USD root {root.GetPath()}")
    body = target
    while body and not body.IsPseudoRoot():
      key = str(body.GetPath())
      if key in body_prims:
        return key, target
      body = body.GetParent()
    # USD also permits a joint endpoint to be a static prim without a
    # RigidBodyAPI. Preserve that explicit endpoint as a stationary part.
    key = str(target.GetPath())
    body_prims[key] = target
    return key, target

  endpoints = {}
  for prim in joint_prims:
    schema = UsdPhysics.Joint(prim)
    endpoints[str(prim.GetPath())] = (
      endpoint(schema.GetBody0Rel()),
      endpoint(schema.GetBody1Rel()),
    )
  for name, prim in body_prims.items():
    robot.add_part(
      Part(
        name,
        transform=_matrix(cache.GetLocalToWorldTransform(prim), meters),
        metadata={
          "usd_type": str(prim.GetTypeName()),
          "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
        },
      )
    )

  for prim in prims:
    if not prim.IsA(UsdGeom.Gprim):
      continue
    owner = prim
    while owner and str(owner.GetPath()) not in robot.parts:
      owner = owner.GetParent()
    if not owner:
      continue  # Environment geometry is not a robot part.
    relative = (
      cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(owner).GetInverse()
    )
    shape = _geometry(prim, _matrix(relative, meters), meters, UsdGeom)
    part = robot.parts[str(owner.GetPath())]
    if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
      part.visuals.append(shape)
    if (
      prim.HasAPI(UsdPhysics.CollisionAPI)
      and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    ):
      part.collisions.append(shape)

  def anchor(body_name, target, position, rotation):
    local = transform(tuple(float(v) * meters for v in position), _quaternion(rotation))
    if target is None:
      return local
    body = body_prims[body_name]
    relative = (
      cache.GetLocalToWorldTransform(target) * cache.GetLocalToWorldTransform(body).GetInverse()
    )
    return multiply(_matrix(relative, meters), local)

  for prim in joint_prims:
    name = str(prim.GetPath())
    schema = UsdPhysics.Joint(prim)
    (parent, target0), (child, target1) = endpoints[name]
    kind, axis, limits = _joint_definition(prim, meters, UsdPhysics)
    joint = kind(
      name,
      parent,
      child,
      axis=axis,
      limits=limits,
      parent_frame=anchor(
        parent, target0, schema.GetLocalPos0Attr().Get(), schema.GetLocalRot0Attr().Get()
      ),
      child_frame=anchor(
        child, target1, schema.GetLocalPos1Attr().Get(), schema.GetLocalRot1Attr().Get()
      ),
      enabled=bool(schema.GetJointEnabledAttr().Get()),
      metadata={
        "usd_type": str(prim.GetTypeName()),
        "exclude_from_articulation": schema.GetExcludeFromArticulationAttr().Get(),
      },
    )
    robot.add_joint(joint)
  return robot


def _joint_definition(prim, meters: float, physics):
  axis_name = prim.GetAttribute("physics:axis").Get() or "X"
  axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
  if axis_name not in axes:
    raise ValueError(f"Invalid USD joint axis {axis_name!r} at {prim.GetPath()}")
  limits = {}
  kind_name = str(prim.GetTypeName())
  if prim.IsA(physics.FixedJoint):
    kind = FixedJoint
  elif prim.IsA(physics.RevoluteJoint) or prim.IsA(physics.PrismaticJoint):
    angular = prim.IsA(physics.RevoluteJoint)
    factor = radians(1) if angular else meters
    lower = _bound(prim.GetAttribute("physics:lowerLimit").Get(), factor)
    upper = _bound(prim.GetAttribute("physics:upperLimit").Get(), factor)
    kind = RevoluteJoint if angular else PrismaticJoint
    if angular and lower is None and upper is None:
      kind = ContinuousJoint
    if lower is not None or upper is not None:
      limits["position"] = JointLimit(lower, upper)
  elif prim.IsA(physics.SphericalJoint):
    kind = SphericalJoint
    for key in ("coneAngle0Limit", "coneAngle1Limit"):
      value = prim.GetAttribute(f"physics:{key}").Get()
      if value is not None and value >= 0:
        limits[key] = JointLimit(upper=radians(value))
  elif prim.IsA(physics.DistanceJoint):
    kind = DistanceJoint
    lower = prim.GetAttribute("physics:minDistance").Get()
    upper = prim.GetAttribute("physics:maxDistance").Get()
    limits["position"] = JointLimit(
      _bound(lower, meters) if lower is not None and lower >= 0 else None,
      _bound(upper, meters) if upper is not None and upper >= 0 else None,
    )
  elif kind_name == "PhysicsJoint":
    kind = D6Joint
    for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ", "distance"):
      if prim.HasAPI(physics.LimitAPI, axis):
        limit = physics.LimitAPI(prim, axis)
        factor = radians(1) if axis.startswith("rot") else meters
        limits[axis] = JointLimit(
          _bound(limit.GetLowAttr().Get(), factor),
          _bound(limit.GetHighAttr().Get(), factor),
        )
  else:
    raise ValueError(f"Unsupported USD joint schema: {kind_name!r}")
  return kind, axes[axis_name], limits


def _geometry(prim, pose, meters: float, geom) -> Geometry:
  kind = str(prim.GetTypeName()).lower()
  shape = Geometry(kind, transform=pose, source=str(prim.GetPath()))
  if prim.IsA(geom.Mesh):
    mesh = geom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
      raise ValueError(f"Incomplete USD mesh at {prim.GetPath()}")
    if sum(counts) != len(indices) or any(n < 3 for n in counts):
      raise ValueError(f"Invalid USD mesh topology at {prim.GetPath()}")
    if any(i < 0 or i >= len(points) for i in indices):
      raise ValueError(f"Invalid USD mesh vertex index at {prim.GetPath()}")
    shape.vertices = tuple(tuple(float(v) * meters for v in p) for p in points)
    faces, offset = [], 0
    for count in counts:
      faces.append(tuple(int(v) for v in indices[offset : offset + count]))
      offset += count
    shape.faces = tuple(faces)
    shape.parameters = {
      "orientation": str(mesh.GetOrientationAttr().Get()),
      "hole_indices": tuple(mesh.GetHoleIndicesAttr().Get() or ()),
    }
  elif prim.IsA(geom.Cube):
    size = float(geom.Cube(prim).GetSizeAttr().Get()) * meters
    shape.kind = "box"
    shape.parameters = {"size": (size, size, size)}
  elif prim.IsA(geom.Sphere):
    shape.parameters = {"radius": float(geom.Sphere(prim).GetRadiusAttr().Get()) * meters}
  elif kind in {"capsule", "cylinder", "cone"}:
    shape.parameters = {
      "radius": float(prim.GetAttribute("radius").Get()) * meters,
      "length": float(prim.GetAttribute("height").Get()) * meters,
      "axis": str(prim.GetAttribute("axis").Get()),
    }
  else:
    raise ValueError(f"Unsupported USD robot geometry {kind!r} at {prim.GetPath()}")
  return shape
