from .joint import Joint


class PrismaticJoint(Joint):
  """Allow translation along one axis within linear limits (1 DoF)."""

  # Future: lock rotation and sideways translation; enforce travel limits.
  kind = "prismatic"
  dof = 1
