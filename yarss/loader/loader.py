"""Public entry point: choose a reader and return the same Robot model."""

from pathlib import Path
from xml.etree import ElementTree as ET

from ..robot import Robot


class Loader:
  """Load URDF, MJCF (.xml/.mjcf), or USD (.usd/.usda/.usdc/.usdz).

  package_paths maps ROS package names to directories for package:// meshes.
  usd_root optionally selects one robot subtree from a larger USD scene.
  """

  def __init__(self, *, package_paths: dict[str, str | Path] | None = None):
    self.package_paths = {
      name: Path(path).expanduser().resolve() for name, path in (package_paths or {}).items()
    }

  def load(self, filename: str | Path, *, usd_root: str | None = None) -> Robot:
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".urdf", ".xml", ".mjcf"}:
      if usd_root is not None:
        raise ValueError("usd_root is only meaningful for USD files")
      root = ET.parse(path).getroot()
      if root.tag == "robot":
        from .urdf import load_urdf

        robot = load_urdf(path, root, self.package_paths)
      elif root.tag == "mujoco":
        from .mjcf import load_mjcf

        robot = load_mjcf(path)
      else:
        raise ValueError(f"Expected <robot> or <mujoco>, found <{root.tag}> in {path}")
    elif suffix in {".usd", ".usda", ".usdc", ".usdz"}:
      from .usd import load_usd

      robot = load_usd(path, usd_root)
    else:
      raise ValueError(f"Unsupported robot file extension {suffix!r}")
    robot.validate()
    return robot


def load_robot(filename: str | Path, *, package_paths=None, usd_root=None) -> Robot:
  """Convenience form of Loader(...).load(...)."""
  return Loader(package_paths=package_paths).load(filename, usd_root=usd_root)
