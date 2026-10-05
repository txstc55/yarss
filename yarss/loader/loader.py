"""Public entry point for loading MJCF robot descriptions."""

from pathlib import Path
from xml.etree import ElementTree as ET

from ..robot import Robot


class Loader:
  """Load MJCF (.xml/.mjcf); other robot formats are not implemented yet."""

  def load(self, filename: str | Path) -> Robot:
    path = Path(filename).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".urdf":
      raise NotImplementedError("URDF loading is not implemented yet; use an MJCF file.")
    if suffix in {".usd", ".usda", ".usdc", ".usdz"}:
      raise NotImplementedError("USD loading is not implemented yet; use an MJCF file.")
    if suffix not in {".xml", ".mjcf"}:
      raise ValueError(f"Unsupported robot file extension {suffix!r}")
    if not path.is_file():
      raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    if root.tag == "robot":
      raise NotImplementedError("URDF loading is not implemented yet; use an MJCF file.")
    if root.tag != "mujoco":
      raise ValueError(f"Expected <mujoco>, found <{root.tag}> in {path}")

    from .mjcf import load_mjcf

    robot = load_mjcf(path)
    robot.validate()
    return robot


def load_robot(filename: str | Path) -> Robot:
  """Convenience form of Loader().load(filename)."""
  return Loader().load(filename)
