from .joint import Joint


class UniversalJoint(Joint):
  """Allow rotation about two intersecting axes (2 DoF)."""

  # Future: keep anchors coincident and block the third rotational direction.
  kind = "universal"
  dof = 2
