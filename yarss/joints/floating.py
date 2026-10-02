from dataclasses import dataclass, field

import numpy as np

from ..transforms import Matrix4
from .joint import Joint


@dataclass
class FloatingJoint(Joint):
  """Allow three translations and three rotations (6 DoF; MJCF 'free')."""

  # Initial world pose of an MJCF free joint, including position and orientation.
  initial_pose: Matrix4 = field(default_factory=lambda: np.eye(4))

  # Future: represent an unconstrained pose with position and orientation.
  kind = "floating"
  dof = 6
