"""Small helpers for reading static poses, not for simulating joint motion.

Matrices use column vectors: world_point = transform @ local_point.
Positions are in meters; quaternions are ordered (w, x, y, z).
"""

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]
Matrix4 = NDArray[np.float64]


def transform(
  position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)
) -> Matrix4:
  """Make an affine matrix, including optional mesh scale."""
  quaternion = np.asarray(quaternion, dtype=float)
  norm = np.linalg.norm(quaternion)
  if norm == 0:
    raise ValueError("A quaternion cannot have zero length")
  w, x, y, z = quaternion / norm
  rotation = np.array(
    [
      (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
      (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
      (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ]
  )
  matrix = np.eye(4)
  matrix[:3, :3] = rotation * np.asarray(scale)
  matrix[:3, 3] = position
  return matrix


def rpy_quaternion(rpy) -> NDArray[np.float64]:
  """URDF fixed-axis roll, pitch, yaw (radians) to a quaternion."""
  half_angles = np.asarray(rpy, dtype=float) / 2
  cr, cp, cy = np.cos(half_angles)
  sr, sp, sy = np.sin(half_angles)
  return np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ]
  )
