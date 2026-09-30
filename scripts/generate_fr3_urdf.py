"""Expand the bundled manufacturer FR3 description without installing ROS.

Run `pip install -e '.[assets]'` before regenerating. Normal loading uses the
already generated URDF and does not require xacro.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import xacro

PACKAGE = Path(__file__).resolve().parents[1] / "example/data/urdf/franka_description"
SOURCE = Path("robots/fr3/fr3.urdf.xacro")
OUTPUT = PACKAGE / "urdf/fr3.urdf"


def main():
  # Resolve ROS package lookups in temporary copies; keep upstream files intact.
  # Xacro only needs the source/configuration files, not the mesh contents.
  with TemporaryDirectory() as directory:
    temporary = Path(directory)
    for source in PACKAGE.rglob("*"):
      if source.suffix not in {".xacro", ".yaml"}:
        continue
      target = temporary / source.relative_to(PACKAGE)
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(source.read_text().replace("$(find franka_description)", str(temporary)))
    document = xacro.process_file(
      str(temporary / SOURCE),
      mappings={"hand": "true", "ee_id": "franka_hand", "with_sc": "false"},
    )
    text = document.toprettyxml(indent="  ")
    text = text.replace(str(temporary / SOURCE), SOURCE.as_posix())
    # Restore package lookups left inside upstream explanatory comments.
    text = text.replace(str(temporary), "$(find franka_description)")
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT.write_text(text)
  print(f"Generated {OUTPUT}")


if __name__ == "__main__":
  main()
