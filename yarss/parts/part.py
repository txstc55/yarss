"""A part is a rigid link/body and the geometry attached to it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..transforms import Matrix4

if TYPE_CHECKING:
  from ..joints import Joint


@dataclass
class Geometry:
  """A mesh reference, inline mesh, or an un-tessellated primitive.

  ``transform`` maps geometry coordinates into the owning part's frame.
  Meshes contain vertices and polygon faces. Primitive parameters use full
  lengths, not half lengths. MJCF mesh scale is baked into the compiled vertices.
  Vertex coordinates, translations, and primitive lengths are in meters.
  """

  kind: str
  transform: Matrix4 = field(default_factory=lambda: np.eye(4))
  parameters: dict[str, object] = field(default_factory=dict)
  mesh_path: Path | None = None
  vertices: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 3)))
  faces: tuple[tuple[int, ...], ...] = ()
  source: str | None = None


@dataclass(eq=False)
class Part:
  """One robot link, including links with no geometry.

  Parts use object identity so they can serve as keys in the robot's joint graph.
  ``transform`` is the part's current world transform, initially the file's default pose.
  The MJCF loader also stores ``local_transform`` relative to the parent body.
  The viewer can edit these poses directly; it does not simulate forces.
  A geometry may appear in both lists when it serves both purposes.
  """

  name: str
  visuals: list[Geometry] = field(default_factory=list)
  collisions: list[Geometry] = field(default_factory=list)
  transform: Matrix4 = field(default_factory=lambda: np.eye(4))
  metadata: dict[str, object] = field(default_factory=dict)
  local_transform: Matrix4 = field(default_factory=lambda: np.eye(4))
  connected_children: list[Part] = field(default_factory=list, repr=False)
  connected_parents: list[Part | None] = field(default_factory=list, repr=False)
  incoming_joints: list[Joint] = field(default_factory=list, repr=False)
  # Filled by Robot.finish_setup(): this part first, then its descendants.
  affected_parts: list[Part] = field(default_factory=list, init=False, repr=False)

  def add_connected_child(self, child: Part) -> None:
    """Record a child once, even when several joints connect the same parts."""
    if child not in self.connected_children:
      self.connected_children.append(child)

  def add_connected_parent(self, parent: Part | None) -> None:
    """Record a parent once; None represents a connection to the world."""
    if parent not in self.connected_parents:
      self.connected_parents.append(parent)

  def add_incoming_joint(self, joint: Joint) -> None:
    """Record an incoming joint once, even when several joints connect the same parts."""
    if joint not in self.incoming_joints:
      self.incoming_joints.append(joint)
