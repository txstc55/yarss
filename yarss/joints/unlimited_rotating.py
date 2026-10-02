from dataclasses import dataclass

from ..transforms import Matrix4, rotation_transform
from .joint import Joint


@dataclass
class UnlimitedRotatingJoint(Joint):
  """Allow unlimited rotation about one axis, as for a wheel (1 DoF)."""

  initial_angle: float = 0.0  # Angle at the loaded pose, in radians.

  # Future: behave as a hinge without angular stops; allow repeated turns.
  kind = "unlimited_rotating"
  dof = 1

  def motion_transform(self) -> Matrix4:
    """Rotate from the initial angle, in radians; the coordinate can span many turns."""
    angle = self.position - self.initial_angle
    return rotation_transform(self.axis, angle)
