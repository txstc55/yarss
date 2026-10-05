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
    return [p for name, p in self.parts.items() if name not in children]

  def connections(self, part_name: str) -> list[Joint]:
    """Return all joints touching a part, including world attachments."""
    if part_name not in self.parts:
      raise KeyError(part_name)
    return [j for j in self.joints.values() if part_name in (j.parent, j.child)]

  def set_joint_position(self, name: str, position: float) -> None:
    """Edit an MJCF hinge/slide coordinate and propagate its child body's pose.

    This is forward kinematics only: no forces, contacts, or coupled joints.
    Angles are in degrees and sliding distances are in meters, just like the sliders.
    """
    if self.metadata.get("format") != "mjcf":
      raise ValueError("Interactive joint poses currently support MJCF robots only")
    joint = self.joints[name]
    if joint.kind not in {"rotating", "unlimited_rotating", "sliding"}:
      raise ValueError(f"Joint {name!r} does not have a supported scalar coordinate")
    if not np.isfinite(position):
      raise ValueError("Joint position must be finite")

    attached = [j for j in self.joints.values() if j.child == joint.child]
    if any(j.kind not in {"fixed", "rotating", "unlimited_rotating", "sliding"} for j in attached):
      raise ValueError("Pose editing requires fixed, hinge, or slide joints on this body")
    joint.position = float(position)
    # Start from the loaded body pose, then apply each joint in source order.
    # Rebuild from absolute coordinates so slider movements never accumulate drift.
    local = attached[0].parent_frame @ np.linalg.inv(attached[0].child_frame)
    for connection in attached:
      anchor = connection.child_frame
      local = local @ anchor @ connection.motion_transform() @ np.linalg.inv(anchor)
    self.parts[joint.child].local_transform = local

    # MJCF bodies form a tree, even when a body contains several joints.
    parents = {j.child: j.parent for j in self.joints.values()}
    children: dict[str | None, set[str]] = {}
    for connection in self.joints.values():
      children.setdefault(connection.parent, set()).add(connection.child)
    pending = [joint.child]
    while pending:
      child = self.parts[pending.pop()]
      parent = parents[child.name]
      parent_world = self.parts[parent].transform if parent is not None else np.eye(4)
      child.transform = parent_world @ child.local_transform
      pending.extend(children.get(child.name, ()))

  def joint_world_frame(self, name: str) -> Matrix4:
    """Locate an MJCF joint's anchor, including earlier joints on the same body."""
    joint = self.joints[name]
    local = joint.parent_frame @ np.linalg.inv(joint.child_frame)
    for connection in self.joints.values():
      if connection is joint:
        break
      if connection.child == joint.child:
        anchor = connection.child_frame
        local = local @ anchor @ connection.motion_transform() @ np.linalg.inv(anchor)
    parent_world = self.parts[joint.parent].transform if joint.parent is not None else np.eye(4)
    return parent_world @ local @ joint.child_frame

  def validate(self) -> None:
    """Reject broken topology and unresolved external geometry files."""
    if not self.parts:
      raise ValueError(f"No robot parts found in {self.source}")
    for joint in self.joints.values():
      if joint.parent == joint.child:
        raise ValueError(f"Joint {joint.name!r} must connect different endpoints")
      for endpoint in (joint.parent, joint.child):
        if endpoint is not None and endpoint not in self.parts:
          raise ValueError(f"Joint {joint.name!r} references missing part {endpoint!r}")
      if joint.mimic and joint.mimic.joint not in self.joints:
        raise ValueError(f"Joint {joint.name!r} mimics missing joint {joint.mimic.joint!r}")
    for part in self.parts.values():
      for geometry in part.visuals + part.collisions:
        if geometry.mesh_path is not None and not geometry.mesh_path.is_file():
          raise FileNotFoundError(f"Part {part.name!r}: {geometry.mesh_path}")
