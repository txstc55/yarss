"""Run from the repository root: python3 example/check_units.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from yarss import Loader, load_robot
from yarss.transforms import rotation_transform, rpy_quaternion, transform


def close(actual, expected):
  # MuJoCo stores compiled mesh vertices as float32.
  np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def world_vertices(part, geometry):
  world = part.transform @ geometry.transform
  return geometry.vertices @ world[:3, :3].T + world[:3, 3]


def check_transforms():
  for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [-2, 3, 5]):
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    for angle in (-450, -90, 0, 30, 90, 180, 720):
      quaternion = np.empty(4)
      expected = np.empty(9)
      mujoco.mju_axisAngle2Quat(quaternion, direction, np.deg2rad(angle))
      mujoco.mju_quat2Mat(expected, quaternion)
      matrix = rotation_transform(np.asarray(axis), angle)
      close(matrix[:3, :3], expected.reshape(3, 3))
      close(matrix[:3, 3], [0, 0, 0])

  rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
  ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
  rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
  for angles, expected in (
    ([90, 0, 0], rx),
    ([0, 90, 0], ry),
    ([0, 0, 90], rz),
    ([90, 90, 90], rz @ ry @ rx),
  ):
    matrix = transform([1.25, -0.2, 0.03], rpy_quaternion(angles))
    close(matrix[:3, :3], expected)
    close(matrix[:3, 3], [1.25, -0.2, 0.03])
  print("Passed: degree arguments in axis-angle and roll/pitch/yaw helpers; meter translations.")


def check_mjcf(folder):
  for unit in ("degree", "radian"):
    factor = 1.0 if unit == "degree" else np.pi / 180
    xml = f"""<mujoco>
  <compiler angle="{unit}"/>
  <asset>
    <mesh name="tetra" scale="0.001 0.001 0.001"
      vertex="0 0 0 100 0 0 0 100 0 0 0 100" face="0 2 1 0 1 3 0 3 2 1 2 3"/>
  </asset>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.1" size="1 2 0.1"/>
    <body name="base" pos="0.1 0.2 0.3">
      <body name="arm" pos="0.3 0 0.2" euler="0 0 {90 * factor}">
        <joint name="hinge" type="hinge" ref="{30 * factor}" springref="{10 * factor}"
          range="{-45 * factor} {75 * factor}" stiffness="12"/>
        <geom name="mesh" type="mesh" mesh="tetra" pos="0.05 0 0"/>
        <body name="finger" pos="0.4 0 0">
          <joint name="slide" type="slide" axis="0 1 0" ref="0.04" range="0 0.1"
            springref="0.02" stiffness="5"/>
          <geom type="box" size="0.01 0.02 0.03"/>
        </body>
      </body>
    </body>
    <body name="wrist" pos="1 0 0">
      <joint name="ball" type="ball" range="0 {60 * factor}" stiffness="7"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="free_body" pos="1 2 3" euler="0 0 {90 * factor}">
      <freejoint name="free"/><geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>"""
    path = folder / f"units_{unit}.xml"
    path.write_text(xml)
    robot = load_robot(path)
    hinge, slide = robot.joints["hinge"], robot.joints["slide"]
    close([hinge.initial_angle, hinge.position], [30, 30])
    close([hinge.limits["position"].lower, hinge.limits["position"].upper], [-45, 75])
    close([slide.initial_distance, slide.position], [0.04, 0.04])
    close([slide.limits["position"].lower, slide.limits["position"].upper], [0, 0.1])
    close(robot.joints["ball"].limits["angle"].upper, 60)
    close(robot.joints["free"].initial_pose[:3, 3], [1, 2, 3])
    close(robot.metadata["world_geometry"][0].parameters["size"], [2, 4, 0.2])
    close(robot.parts["finger"].visuals[0].parameters["size"], [0.02, 0.04, 0.06])
    points = world_vertices(robot.parts["arm"], robot.parts["arm"].visuals[0])
    close(points.min(axis=0), [0.3, 0.25, 0.5])
    close(points.max(axis=0), [0.4, 0.35, 0.6])

    # Independently check that degree-based stiffness still produces native torque.
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("hinge").id]] = np.deg2rad(40)
    data.qpos[model.jnt_qposadr[model.joint("slide").id]] = 0.07
    mujoco.mj_forward(model, data)
    for name, displacement in (("hinge", 40 - 10), ("slide", 0.07 - 0.02)):
      dof = model.jnt_dofadr[model.joint(name).id]
      stiffness = robot.joints[name].metadata["stiffness"]
      close(-stiffness * displacement, data.qfrc_passive[dof])
    close(robot.joints["ball"].metadata["stiffness"], 7 * np.pi / 180)
  print("Passed: MJCF angle declarations, states, limits, mesh scales, lengths, and spring units.")


def check_loader(folder):
  """Keep unsupported formats explicit, including URDF content in XML files."""
  for load in (load_robot, Loader().load):
    for suffix in (".urdf", ".URDF", ".usd", ".usda", ".usdc", ".usdz", ".USD"):
      path = folder / ("unsupported" + suffix)
      expected = "URDF" if suffix.lower() == ".urdf" else "USD"
      try:
        load(path)
      except NotImplementedError as error:
        assert str(error) == f"{expected} loading is not implemented yet; use an MJCF file."
      else:
        raise AssertionError(f"{suffix} must raise NotImplementedError")

    path = folder / "unsupported.xml"
    path.write_text('<robot name="unsupported"/>')
    try:
      load(path)
    except NotImplementedError as error:
      assert "URDF loading is not implemented" in str(error)
    else:
      raise AssertionError("A <robot> XML file must raise NotImplementedError")

    # The .mjcf extension must still resolve assets and included files locally.
    path = folder / "robot.mjcf"
    path.write_text('<mujoco><include file="body.xml"/></mujoco>')
    (folder / "body.xml").write_text(
      '<mujoco><worldbody><body name="base"><geom size="0.1"/></body></worldbody></mujoco>'
    )
    assert "base" in load(path).parts
  print("Passed: MJCF file includes and explicit not-implemented errors for other formats.")


def main():
  check_transforms()
  with TemporaryDirectory() as folder:
    folder = Path(folder)
    check_mjcf(folder)
    check_loader(folder)
  path = Path(__file__).parent / "data/mjcf/franka_fr3/scene.xml"
  robot = load_robot(path)
  assert len(robot.parts) == 12
  assert len(robot.joints) == 12
  assert robot.metadata["world_geometry"]
  print("Passed: bundled FR3 scene includes the full arm, gripper, and world geometry.")


if __name__ == "__main__":
  main()
