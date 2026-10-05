"""A robot's parts, joints, and editable poses."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..joints import Joint
from ..parts import Part
from ..transforms import Matrix4


@dataclass
class Robot:
  """Parts are graph nodes; joints are edges, in source order.

  Several MJCF joints can connect the same pair of bodies and retain their
  source order when composing motion.
  """

  name: str
  source: Path
  parts: dict[str, Part] = field(default_factory=dict)
  joints: dict[str, Joint] = field(default_factory=dict)
  metadata: dict[str, object] = field(default_factory=dict)
  _setup_finished: bool = field(default=False, init=False, repr=False)

  def add_part(self, part: Part) -> None:
    if not part.name or part.name in self.parts:
      raise ValueError(f"Empty or duplicate part name: {part.name!r}")
    self.parts[part.name] = part
    self._setup_finished = False

  def add_joint(self, joint: Joint) -> None:
    if not joint.name or joint.name in self.joints:
      raise ValueError(f"Empty or duplicate joint name: {joint.name!r}")
    self.joints[joint.name] = joint
    self._setup_finished = False

  def finish_setup(self) -> None:
    """Cache each part and its descendants in parent-before-child update order.

    Call after adding all parts and joints, and again if their connections change.
    The loader calls this automatically.
    """
    self._setup_finished = False
    self.validate()
    for part in self.parts.values():
      if len(part.connected_parents) > 1:
        raise ValueError(f"Part {part.name!r} has multiple parents; pose editing requires a tree")
      affected = []
      seen = set()
      pending = [part]
      while pending:
        child = pending.pop()
        if child in seen:
          raise ValueError(f"Repeated part {child.name!r}; pose editing requires a tree")
        seen.add(child)
        affected.append(child)
        # Reverse the stack input to preserve the children's definition order.
        pending.extend(reversed(child.connected_children))
      part.affected_parts = affected
    self._setup_finished = True

  @property
  def roots(self) -> list[Part]:
    """Parts without an incoming part-to-part edge; loops may have none."""
    children = {j.child for j in self.joints.values() if j.parent is not None}
    return [part for part in self.parts.values() if part not in children]

  def connections(self, part_name: str) -> list[Joint]:
    """Return all joints touching a part, including world attachments."""
    part = self.parts[part_name]
    return [j for j in self.joints.values() if j.parent is part or j.child is part]

  def set_joint_value(self, joint: Joint, value: float) -> list[Part]:
    """
    This is forward kinematics only: no forces, contacts, or coupled joints.
    Angles are in degrees and sliding distances are in meters, just like the sliders.
    Return the cached affected-part list so the viewer can update only their geometry.
    """
    if not np.isfinite(value):
      raise ValueError("Joint value must be finite")
    attached = joint.child.incoming_joints
    joint.value = float(value)
    # Start from the loaded body pose, then apply each joint in source order.
    # Rebuild from absolute coordinates so slider movements never accumulate drift.
    local = joint.parent_frame @ joint.child_frame_inverse
    for connection in attached:
      anchor = connection.child_frame
      local = local @ anchor @ connection.motion_transform() @ connection.child_frame_inverse
    joint.child.local_transform = local

    # Setup already put every parent before its children in this list.
    for child in joint.child.affected_parts:
      parent = child.connected_parents[0]
      child.transform = (
        parent.transform @ child.local_transform
        if parent is not None
        else child.local_transform.copy()
      )
    return joint.child.affected_parts

  def joint_world_frame(self, name: str) -> Matrix4:
    """Locate an MJCF joint's anchor, including earlier joints on the same body."""
    joint = self.joints[name]
    local = joint.parent_frame @ joint.child_frame_inverse
    for connection in joint.child.incoming_joints:
      if connection is joint:
        break
      anchor = connection.child_frame
      local = local @ anchor @ connection.motion_transform() @ connection.child_frame_inverse
    parent_world = joint.parent.transform if joint.parent is not None else np.eye(4)
    return parent_world @ local @ joint.child_frame

  def validate(self) -> None:
    """Reject broken topology and unresolved external geometry files."""
    if not self.parts:
      raise ValueError(f"No robot parts found in {self.source}")
    for joint in self.joints.values():
      if joint.parent is joint.child:
        raise ValueError(f"Joint {joint.name!r} must connect different endpoints")
      for endpoint in (joint.parent, joint.child):
        if endpoint is not None and self.parts.get(endpoint.name) is not endpoint:
          raise ValueError(f"Joint {joint.name!r} references unregistered part {endpoint.name!r}")
      if joint.mimic and joint.mimic.joint not in self.joints:
        raise ValueError(f"Joint {joint.name!r} mimics missing joint {joint.mimic.joint!r}")
    for part in self.parts.values():
      for geometry in part.visuals + part.collisions:
        if geometry.mesh_path is not None and not geometry.mesh_path.is_file():
          raise FileNotFoundError(f"Part {part.name!r}: {geometry.mesh_path}")
