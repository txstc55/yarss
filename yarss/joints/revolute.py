from .joint import Joint


class RevoluteJoint(Joint):
  """Allow rotation about one axis within angular limits (1 DoF)."""

  # Future: constrain translation and off-axis rotation; enforce angle limits.
  kind = "revolute"
  dof = 1
