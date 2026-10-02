from dataclasses import dataclass

import numpy as np

from ..transforms import Matrix4
from .joint import Joint


@dataclass
class SlidingJoint(Joint):
  """Allow translation along one axis within linear limits (1 DoF)."""

  initial_distance: float = 0.0  # Distance at the loaded pose, in meters.

  # Future: lock rotation and sideways translation; enforce travel limits.
  kind = "sliding"
  dof = 1

  def motion_transform(self) -> Matrix4:
    """Translate along the axis by the change from the initial distance, in meters."""
    distance = self.position - self.initial_distance
    axis = self.axis / np.linalg.norm(self.axis)
    matrix = np.eye(4)
    matrix[:3, 3] = axis * distance
    return matrix
