from .joint import Joint


class PlanarJoint(Joint):
  """Allow two translations and rotation in a plane (3 DoF)."""

  # Future: axis is the plane normal; lock normal translation and tilting.
  kind = "planar"
  dof = 3
