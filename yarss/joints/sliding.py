import numpy as np

from ..transforms import Matrix4
from .joint import Joint


class SlidingJoint(Joint):
  """Allow translation along one axis within linear limits (1 DoF)."""

  # Future: lock rotation and sideways translation; enforce travel limits.
  kind = "sliding"
  dof = 1

  def motion_transform(self) -> Matrix4:
    """Translate along the axis by the change from the initial distance, in meters."""
    delta = self.value - self.initial_value
    axis = self.axis / np.linalg.norm(self.axis)
    matrix = np.eye(4)
    matrix[:3, 3] = axis * delta
    return matrix
