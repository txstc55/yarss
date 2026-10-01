"""A part is a rigid link/body and the geometry attached to it."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..transforms import Matrix4


@dataclass
class Geometry:
  """A mesh reference, inline mesh, or an un-tessellated primitive.

  ``transform`` maps geometry coordinates into the owning part's frame.
  Meshes contain vertices and polygon faces; external URDF meshes also keep
  their resolved file path. Primitive parameters use full lengths, not half lengths.
  """

  kind: str
  transform: Matrix4 = field(default_factory=lambda: np.eye(4))
  parameters: dict[str, object] = field(default_factory=dict)
  mesh_path: Path | None = None
  vertices: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 3)))
  faces: tuple[tuple[int, ...], ...] = ()
  source: str | None = None


@dataclass
class Part:
  """One robot link, including links with no geometry.

  ``transform`` is the part's world transform at the file's default pose.
  USD scale is retained in this matrix. No runtime state is simulated here.
  A geometry may appear in both lists when it serves both purposes.
  """

  name: str
  visuals: list[Geometry] = field(default_factory=list)
  collisions: list[Geometry] = field(default_factory=list)
  transform: Matrix4 = field(default_factory=lambda: np.eye(4))
  metadata: dict[str, object] = field(default_factory=dict)
