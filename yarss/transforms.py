"""Small helpers for reading static poses, not for simulating joint motion.

Matrices use column vectors: world_point = transform @ local_point.
Positions are in meters; quaternions are ordered (w, x, y, z).
"""

from math import cos, sin, sqrt

Vector3 = tuple[float, float, float]
Matrix4 = tuple[tuple[float, ...], ...]

IDENTITY: Matrix4 = (
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 1.0),
)


def multiply(a: Matrix4, b: Matrix4) -> Matrix4:
  return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4))


def transform(
  position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)
) -> Matrix4:
  """Make an affine matrix, including optional mesh scale."""
  norm = sqrt(sum(float(v) ** 2 for v in quaternion))
  if norm == 0:
    raise ValueError("A quaternion cannot have zero length")
  w, x, y, z = (float(v) / norm for v in quaternion)
  rotation = (
    (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
    (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
    (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
  )
  return tuple(
    tuple(rotation[i][j] * float(scale[j]) for j in range(3)) + (float(position[i]),)
    for i in range(3)
  ) + ((0.0, 0.0, 0.0, 1.0),)


def rpy_quaternion(rpy: Vector3) -> tuple[float, float, float, float]:
  """URDF fixed-axis roll, pitch, yaw (radians) to a quaternion."""
  cr, cp, cy = (cos(v / 2) for v in rpy)
  sr, sp, sy = (sin(v / 2) for v in rpy)
  return (
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  )
