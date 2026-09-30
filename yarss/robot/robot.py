"""The shared representation returned by all three loaders."""

from dataclasses import dataclass, field
from pathlib import Path

from ..joints import Joint
from ..parts import Part


@dataclass
class Robot:
  """Parts are graph nodes; joints are edges, in source order.

  A graph permits USD closed loops and several MJCF joints between the
  same pair of bodies. It does not assume every robot is a URDF tree.
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
