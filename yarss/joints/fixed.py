import numpy as np

from ..transforms import Matrix4
from .joint import Joint


class FixedJoint(Joint):
  """Keep the two anchors at the same position and orientation (0 DoF)."""

  # Future: constrain all three translations and all three rotations.
  kind = "fixed"
  dof = 0

  def motion_transform(self) -> Matrix4:
    """A fixed joint adds no movement between its attachment frames."""
    return np.eye(4)
