from .joint import Joint


class RotatingJoint(Joint):
  """Allow rotation about one axis within angular limits (1 DoF)."""

  # Future: constrain translation and off-axis rotation; enforce angle limits.
  kind = "rotating"
  dof = 1
