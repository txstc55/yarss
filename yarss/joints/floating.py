from dataclasses import dataclass, field

import numpy as np

from ..transforms import Matrix4
from .joint import Joint


@dataclass
class FloatingJoint(Joint):
  """Allow three translations and three rotations (6 DoF; MJCF 'free')."""

  # Current and loaded world poses, each a 4x4 transform with meter translations.
  value: Matrix4 = field(default_factory=lambda: np.eye(4))
  initial_value: Matrix4 = field(default_factory=lambda: np.eye(4))

  # Future: represent an unconstrained pose with position and orientation.
  kind = "floating"
  dof = 6
