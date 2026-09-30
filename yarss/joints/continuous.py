from .joint import Joint


class ContinuousJoint(Joint):
  """Allow unlimited rotation about one axis, as for a wheel (1 DoF)."""

  # Future: behave as a hinge without angular stops; allow repeated turns.
  kind = "continuous"
  dof = 1
