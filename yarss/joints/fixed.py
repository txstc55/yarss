from .joint import Joint


class FixedJoint(Joint):
  """Keep the two anchors at the same position and orientation (0 DoF)."""

  # Future: constrain all three translations and all three rotations.
  kind = "fixed"
  dof = 0
