"""NumPy helpers for pose matrices and quaternions.

Matrices use column vectors: world_point = transform @ local_point.
Positions are in meters and angle arguments are in degrees.
Quaternions are dimensionless and ordered (w, x, y, z).
"""

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]
Matrix4 = NDArray[np.float64]


def rotation_transform(axis: Vector3, angle: float) -> Matrix4:
  """Build a 4x4 rotation directly from an axis and an angle in degrees."""
  axis = np.asarray(axis, dtype=float)
  length = np.linalg.norm(axis)
  if length == 0:
    raise ValueError("A rotation axis cannot have zero length")
  axis = axis / length
  x, y, z = axis
  # cross @ vector gives the cross product of the axis with that vector.
  cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
  # NumPy's trigonometric functions take radians; joint state stays in degrees.
  radians = np.deg2rad(angle)
  cosine, sine = np.cos(radians), np.sin(radians)
  # Rodrigues' formula: rotate around the axis while leaving the axis itself fixed.
  matrix = np.eye(4)
  matrix[:3, :3] = cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * cross
  return matrix


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
  """Fixed-axis roll, pitch, yaw (degrees) to a quaternion."""
  half_angles = np.deg2rad(np.asarray(rpy, dtype=float)) / 2
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
