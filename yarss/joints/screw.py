from .joint import Joint


class ScrewJoint(Joint):
  """Couple rotation and translation by a screw pitch (1 DoF)."""

  # Future: translation = pitch * angle / (2*pi), with pitch in meters/turn.
  kind = "screw"
  dof = 1
