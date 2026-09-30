"""Joint definitions only; constraint forces and motion are future work."""

from dataclasses import dataclass, field
from typing import ClassVar

from ..transforms import IDENTITY, Matrix4, Vector3


@dataclass
class JointLimit:
  """Meters or radians; None means that bound was not specified.

  For USD D6 joints, lower > upper means the axis is locked.
  Effort and velocity are retained when the source specifies them.
  """

  lower: float | None = None
  upper: float | None = None
  effort: float | None = None
  velocity: float | None = None


@dataclass
class Mimic:
  """Future behavior: q = multiplier * referenced_joint.q + offset."""

  joint: str
  multiplier: float = 1.0
  offset: float = 0.0


@dataclass
class Joint:
  """Connect two named parts. A None endpoint means the world.

  Each anchor matrix maps the joint frame into that endpoint's part frame.
  Axis is expressed in the joint frame, not in world coordinates.
  Scalar limits use the key 'position'; D6 limits use transX/Y/Z, rotX/Y/Z.
  Multiple MJCF joints on one body share endpoints and retain source order.

  Future solvers should use these definitions to construct constraints,
  apply limits/drives, and update part poses. Loading does none of that.
  """

  name: str
  parent: str | None
  child: str | None
  parent_frame: Matrix4 = IDENTITY
  child_frame: Matrix4 = IDENTITY
  axis: Vector3 = (1.0, 0.0, 0.0)
  limits: dict[str, JointLimit] = field(default_factory=dict)
  mimic: Mimic | None = None
  enabled: bool = True
  metadata: dict[str, object] = field(default_factory=dict)

  kind: ClassVar[str] = "joint"
  dof: ClassVar[int | None] = None
