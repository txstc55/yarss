"""Preview MJCF geometry and edit joint poses after running `./yarss/install.sh`."""

import argparse
from pathlib import Path

from yarss import load_robot
from yarss.viewer import Viewer


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "mjcf",
    nargs="?",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "data/mjcf/franka_fr3/fr3.xml",
    help="MJCF file to load (defaults to the bundled FR3)",
  )
  args = parser.parse_args()

  robot = load_robot(args.mjcf)
  viewer = Viewer()
  viewer.add_robot(robot)
  print(
    f"Showing {len(viewer.meshes)} visual meshes and "
    f"{len(viewer.collision_meshes)} collision shapes from {args.mjcf}"
  )
  viewer.show()


if __name__ == "__main__":
  main()
