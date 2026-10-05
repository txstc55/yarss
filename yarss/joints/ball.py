from dataclasses import dataclass, field

import numpy as np

from ..transforms import Matrix3
from .joint import Joint


@dataclass
class BallJoint(Joint):
  """Allow rotation in all directions about one point (3 DoF; 'ball')."""

  # Current and loaded joint orientations, each a 3x3 rotation matrix.
  value: Matrix3 = field(default_factory=lambda: np.eye(3))
  initial_value: Matrix3 = field(default_factory=lambda: np.eye(3))

  # Future: keep anchors coincident; apply swing/twist or cone limits if given.
  kind = "ball"
  dof = 3
