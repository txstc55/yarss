from .joint import Joint


class UnlimitedRotatingJoint(Joint):
  """Allow unlimited rotation about one axis, as for a wheel (1 DoF)."""

  # Future: behave as a hinge without angular stops; allow repeated turns.
  kind = "unlimited_rotating"
  dof = 1
