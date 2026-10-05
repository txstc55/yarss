from ..transforms import Matrix4, rotation_transform
from .joint import Joint


class RotatingJoint(Joint):
  """Allow rotation about one axis within angular limits (1 DoF)."""

  # Future: constrain translation and off-axis rotation; enforce angle limits.
  kind = "rotating"
  dof = 1

  def motion_transform(self) -> Matrix4:
    """Rotate about the axis by the change from the initial angle, in degrees."""
    delta = self.value - self.initial_value
    return rotation_transform(self.axis, delta)
