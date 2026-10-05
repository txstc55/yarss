from ..transforms import Matrix4, rotation_transform
from .joint import Joint


class UnlimitedRotatingJoint(Joint):
  """Allow unlimited rotation about one axis, as for a wheel (1 DoF)."""

  # Future: behave as a hinge without angular stops; allow repeated turns.
  kind = "unlimited_rotating"
  dof = 1

  def motion_transform(self) -> Matrix4:
    """Rotate from the initial angle, in degrees; the coordinate can span many turns."""
    delta = self.value - self.initial_value
    return rotation_transform(self.axis, delta)
