from dataclasses import dataclass

from ..transforms import Matrix4, rotation_transform
from .joint import Joint


@dataclass
class RotatingJoint(Joint):
  """Allow rotation about one axis within angular limits (1 DoF)."""

  initial_angle: float = 0.0  # Angle at the loaded pose, in degrees.

  # Future: constrain translation and off-axis rotation; enforce angle limits.
  kind = "rotating"
  dof = 1

  def motion_transform(self) -> Matrix4:
    """Rotate about the axis by the change from the initial angle, in degrees."""
    angle = self.position - self.initial_angle
    return rotation_transform(self.axis, angle)
