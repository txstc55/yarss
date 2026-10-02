from .joint import Joint


class SlidingJoint(Joint):
  """Allow translation along one axis within linear limits (1 DoF)."""

  # Future: lock rotation and sideways translation; enforce travel limits.
  kind = "sliding"
  dof = 1
