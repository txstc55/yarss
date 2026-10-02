from dataclasses import dataclass, field

import numpy as np

from ..transforms import Matrix3
from .joint import Joint


@dataclass
class BallJoint(Joint):
  """Allow rotation in all directions about one point (3 DoF; 'ball')."""

  # Initial joint orientation relative to the loaded attachment frame.
  initial_orientation: Matrix3 = field(default_factory=lambda: np.eye(3))

  # Future: keep anchors coincident; apply swing/twist or cone limits if given.
  kind = "ball"
  dof = 3
