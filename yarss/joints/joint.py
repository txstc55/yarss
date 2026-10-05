"""Joint definitions and scalar pose editing; constraint forces are future work."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..parts import Part
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
  """Connect two Part objects. A None parent means the world.

  Each anchor matrix maps the joint frame into that endpoint's part frame.
  ``child_frame`` is fixed after construction; its inverse is computed once.
  Axis is expressed in the joint frame, not in world coordinates.
  Scalar limits use the key 'value'; ball cone limits use 'angle'.
  D6 limits use transX/Y/Z, rotX/Y/Z.
  Multiple MJCF joints on one body share endpoints and retain source order.
  ``value`` is the current joint state; ``initial_value`` is the loaded state.
  Rotating and sliding joints use scalars in degrees and meters, respectively.
  Ball and floating joints use rotation and pose matrices.

  Future solvers should use these definitions to construct constraints,
  apply limits/drives, and update part poses. Loading does none of that.
  """

  name: str
  parent: Part | None  # the parent it is attached to
  child: Part  # the child it is attached to
  parent_frame: Matrix4 = field(
    default_factory=lambda: np.eye(4)
  )  # where this joint is relative to parent
  child_frame: Matrix4 = field(
    default_factory=lambda: np.eye(4)
  )  # where this joint is relative to child
  child_frame_inverse: Matrix4 = field(init=False, repr=False)
  axis: Vector3 = field(
    default_factory=lambda: np.array([1.0, 0.0, 0.0])
  )  # the axis along which this joint moves, expressed in the joint frame
  limits: dict[str, JointLimit] = field(default_factory=dict)  # the limit for this joint
  mimic: Mimic | None = None
  enabled: bool = True
  metadata: dict[str, object] = field(default_factory=dict)
  value: float | np.ndarray = 0.0
  initial_value: float | np.ndarray = 0.0

  kind: ClassVar[str] = "joint"
  dof: ClassVar[int | None] = None

  def __post_init__(self) -> None:
    # Keep a fixed copy so edits to the input matrix cannot invalidate the inverse.
    self.child_frame = np.array(self.child_frame, dtype=float, copy=True)
    self.child_frame.setflags(write=False)
    self.child_frame_inverse = np.linalg.inv(self.child_frame)
    self.child_frame_inverse.setflags(write=False)

    # Store the hierarchy once; incoming joints retain their definition order.
    if self.parent is not None:
      self.parent.add_connected_child(self.child)
    self.child.add_connected_parent(self.parent)
    self.child.add_incoming_joint(self)

  def motion_transform(self) -> Matrix4:
    """Subclasses define motion from the loaded pose, in joint coordinates."""
    raise NotImplementedError(f"Pose editing is not implemented for {self.kind} joints")
