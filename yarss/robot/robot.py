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

  def add_part(self, part: Part) -> None:
    if not part.name or part.name in self.parts:
      raise ValueError(f"Empty or duplicate part name: {part.name!r}")
    self.parts[part.name] = part

  def add_joint(self, joint: Joint) -> None:
    if not joint.name or joint.name in self.joints:
      raise ValueError(f"Empty or duplicate joint name: {joint.name!r}")
    self.joints[joint.name] = joint

  @property
  def roots(self) -> list[Part]:
    """Parts without an incoming part-to-part edge; loops may have none."""
    children = {j.child for j in self.joints.values() if j.parent is not None}
    return [part for part in self.parts.values() if part not in children]

  def connections(self, part_name: str) -> list[Joint]:
    """Return all joints touching a part, including world attachments."""
    part = self.parts[part_name]
    return [j for j in self.joints.values() if j.parent is part or j.child is part]

  def set_joint_value(self, name: str, value: float) -> set[Part]:
    """
    This is forward kinematics only: no forces, contacts, or coupled joints.
    Angles are in degrees and sliding distances are in meters, just like the sliders.
    Return the affected parts so the viewer can update only their geometry.
    """
    if self.metadata.get("format") != "mjcf":
      raise ValueError("Interactive joint poses currently support MJCF robots only")
    joint = self.joints[name]
    if joint.kind not in {"rotating", "unlimited_rotating", "sliding"}:
      raise ValueError(f"Joint {name!r} does not have a supported scalar coordinate")
    if not np.isfinite(value):
      raise ValueError("Joint value must be finite")

    attached = joint.child.incoming_joints
    if any(j.kind not in {"fixed", "rotating", "unlimited_rotating", "sliding"} for j in attached):
      raise ValueError("Pose editing requires fixed, hinge, or slide joints on this body")
    joint.value = float(value)
    # Start from the loaded body pose, then apply each joint in source order.
    # Rebuild from absolute coordinates so slider movements never accumulate drift.
    local = attached[0].parent_frame @ attached[0].child_frame_inverse
    for connection in attached:
      anchor = connection.child_frame
      local = local @ anchor @ connection.motion_transform() @ connection.child_frame_inverse
    joint.child.local_transform = local

    # MJCF has one parent per body; several joints can share that same parent.
    updated: set[Part] = set()
    pending = [joint.child]
    while pending:
      child = pending.pop()
      parent = child.connected_parents[0]
      child.transform = (
        parent.transform @ child.local_transform
        if parent is not None
        else child.local_transform.copy()
      )
      updated.add(child)
      pending.extend(child.connected_children)
    return updated

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
