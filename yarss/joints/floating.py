from .joint import Joint


class FloatingJoint(Joint):
  """Allow three translations and three rotations (6 DoF; MJCF 'free')."""

  # Future: represent an unconstrained pose with position and orientation.
  kind = "floating"
  dof = 6
