"""Joint definitions and scalar pose editing; constraint forces are future work."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..transforms import Matrix4, Vector3


@dataclass
class JointLimit:
  """Bounds in meters or degrees; None means that limit was not specified.

  For D6 joint placeholders, lower > upper means the axis is locked.
  Effort is a force limit in N or a torque limit in N*m.
  Velocity is a speed limit in m/s or degrees/s, when supplied by the source.
  """

  lower: float | None = None
  upper: float | None = None
  effort: float | None = None
  velocity: float | None = None


@dataclass
class Mimic:
  """Future behavior: q = multiplier * referenced_joint.q + offset.

  Offset uses the driven joint's units: meters or degrees.
  """

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
  ``position`` is the current scalar coordinate, in degrees or meters.

  Future solvers should use these definitions to construct constraints,
  apply limits/drives, and update part poses. Loading does none of that.
  """

  name: str
  parent: str | None  # the parent it is attached to
  child: str | None  # the child it is attached to
  parent_frame: Matrix4 = field(
    default_factory=lambda: np.eye(4)
  )  # where this joint is relative to parent
  child_frame: Matrix4 = field(
    default_factory=lambda: np.eye(4)
  )  # where this joint is relative to child
  axis: Vector3 = field(
    default_factory=lambda: np.array([1.0, 0.0, 0.0])
  )  # the axis along which this joint moves, expressed in the joint frame
  limits: dict[str, JointLimit] = field(default_factory=dict)  # the limit for this joint
  mimic: Mimic | None = None
  enabled: bool = True
  metadata: dict[str, object] = field(default_factory=dict)
  position: float = 0.0  # this is actually the joint's current value, like the angle, or the displacement along the axis

  kind: ClassVar[str] = "joint"
  dof: ClassVar[int | None] = None

  def motion_transform(self) -> Matrix4:
    """Subclasses define motion from the loaded pose, in joint coordinates."""
    raise NotImplementedError(f"Pose editing is not implemented for {self.kind} joints")
