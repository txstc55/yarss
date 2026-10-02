"""Run from the repository root: python3 example/viewer/check_joint_controls.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np
import pyvista as pv

from yarss import load_robot
from yarss.joints import FixedJoint, RotatingJoint, SlidingJoint, UnlimitedRotatingJoint
from yarss.viewer import Viewer


def check_joint_motions():
  """Check a diagonal, non-unit axis with known movements and nonzero initial values."""
  axis = np.array([0.0, 1.0, 1.0])
  diagonal = 1 / np.sqrt(2)
  for kind in (RotatingJoint, UnlimitedRotatingJoint):
    joint = kind("turn", "parent", "child", axis=axis, initial_angle=0.3)
    joint.position = 0.3
    np.testing.assert_allclose(joint.motion_transform(), np.eye(4), atol=1e-12)
    for angle in (np.pi / 2, np.pi / 2 + 4 * np.pi):
      joint.position = 0.3 + angle
      motion = joint.motion_transform()
      np.testing.assert_allclose(motion[:3, 0], [0, diagonal, -diagonal], atol=1e-12)
      np.testing.assert_allclose(motion[:3, :3] @ axis, axis, atol=1e-12)
      np.testing.assert_allclose(motion.T @ motion, np.eye(4), atol=1e-12)
      np.testing.assert_allclose(np.linalg.det(motion[:3, :3]), 1, atol=1e-12)

  joint = SlidingJoint(
    "slide",
    "parent",
    "child",
    axis=axis,
    position=0.14,
    initial_distance=0.04,
  )
  expected = np.eye(4)
  expected[:3, 3] = [0, 0.1 * diagonal, 0.1 * diagonal]
  np.testing.assert_allclose(joint.motion_transform(), expected, atol=1e-12)
  np.testing.assert_array_equal(
    FixedJoint("fixed", "parent", "child").motion_transform(), np.eye(4)
  )
  np.testing.assert_array_equal(axis, [0, 1, 1])


def check_initial_states():
  """All MJCF joint types retain their initial state in explicit fields."""
  xml = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="arm">
      <joint name="angle" type="hinge" ref="0.3" range="-1 1"/>
      <geom size="0.1"/>
    </body>
    <body name="wheel">
      <joint name="spin" type="hinge" ref="0.7"/>
      <geom size="0.1"/>
    </body>
    <body name="finger">
      <joint name="distance" type="slide" ref="0.04"/>
      <geom size="0.1"/>
    </body>
    <body name="wrist" euler="0.2 0.3 0.4">
      <joint name="orientation" type="ball"/>
      <geom size="0.1"/>
    </body>
    <body name="free_body" pos="0.2 0.3 0.4" euler="0.1 0.2 0.3">
      <freejoint name="pose"/>
      <geom size="0.1"/>
    </body>
  </worldbody>
</mujoco>"""
  with TemporaryDirectory() as folder:
    path = Path(folder) / "initial_states.xml"
    path.write_text(xml)
    robot = load_robot(path)
    assert robot.joints["angle"].initial_angle == 0.3
    assert robot.joints["spin"].initial_angle == 0.7
    assert robot.joints["distance"].initial_distance == 0.04
    np.testing.assert_array_equal(robot.joints["orientation"].initial_orientation, np.eye(3))
    np.testing.assert_allclose(
      robot.joints["pose"].initial_pose, robot.parts["free_body"].transform, atol=1e-12
    )
    for name, value in (("angle", 0.6), ("spin", 2.0), ("distance", 0.02)):
      joint = robot.joints[name]
      initial = joint.initial_distance if joint.kind == "sliding" else joint.initial_angle
      assert joint.position == initial
      robot.set_joint_position(name, value)
      assert (joint.initial_distance if joint.kind == "sliding" else joint.initial_angle) == initial
    for joint in robot.joints.values():
      assert set(joint.metadata) == {"order_in_body", "stiffness"}


def check_poses(robot, model, data):
  """Use native MuJoCo kinematics as an independent reference, without stepping physics."""
  mujoco.mj_kinematics(model, data)
  for body_id in range(1, model.nbody):
    part = robot.parts[model.body(body_id).name]
    np.testing.assert_allclose(part.transform[:3, 3], data.xpos[body_id], atol=1e-12)
    np.testing.assert_allclose(part.transform[:3, :3], data.xmat[body_id].reshape(3, 3), atol=1e-12)
    parent_id = model.body_parentid[body_id]
    parent = robot.parts[model.body(parent_id).name].transform if parent_id else np.eye(4)
    np.testing.assert_allclose(parent @ part.local_transform, part.transform, atol=1e-12)
  for joint_id in range(model.njnt):
    joint = robot.joints[model.joint(joint_id).name]
    frame = robot.joint_world_frame(joint.name)
    np.testing.assert_allclose(frame[:3, 3], data.xanchor[joint_id], atol=1e-12)
    np.testing.assert_allclose(frame[:3, :3] @ joint.axis, data.xaxis[joint_id], atol=1e-12)


def check_meshes(viewer, model, data):
  for prefix, meshes, group in (
    ("mesh", viewer.meshes, 2),
    ("collision", viewer.collision_meshes, 3),
  ):
    geom_ids = np.flatnonzero(model.geom_group == group)
    for index, (mesh, geom_id) in enumerate(zip(meshes, geom_ids, strict=True)):
      if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start : start + count].astype(float)
        world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        np.testing.assert_allclose(mesh.points, world, atol=1e-7)
      else:
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        np.testing.assert_allclose(mesh.center, data.geom_xpos[geom_id], atol=1e-7)
      actor_mesh = viewer.plotter.actors[f"{prefix}_{index}"].mapper.dataset
      np.testing.assert_array_equal(actor_mesh.points, mesh.points)

  expected_labels = []
  for index, (_, joint, mesh, length) in enumerate(viewer._joint_axes):
    joint_id = model.joint(joint.name).id
    expected_labels.append(data.xanchor[joint_id] + length * data.xaxis[joint_id])
    actor_mesh = viewer.plotter.actors[f"joint_axis_{index}"].mapper.dataset
    np.testing.assert_array_equal(actor_mesh.points, mesh.points)
  hierarchy = viewer.plotter.actors["joint_names-labels"].GetMapper().GetInputAlgorithm()
  label_points = pv.wrap(hierarchy.GetInputDataObject(0, 0)).points
  np.testing.assert_allclose(label_points, expected_labels, atol=1e-12)


def check_shared_body():
  """Offset anchors and several joints on a body must compose in source order."""
  xml = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0.1 0.2 0.3" euler="0.2 0.3 0.4">
      <body name="moving" pos="0.3 0 0.2" euler="0.1 -0.2 0.3">
        <joint name="hinge" type="hinge" pos="0.04 0.05 0.06" axis="1 2 3" ref="0.2"/>
        <joint name="slide" type="slide" axis="0 1 0" ref="0.03"/>
        <geom type="box" size="0.1 0.1 0.1"/>
        <body name="tip" pos="0 0 0.3"><geom type="sphere" size="0.02"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
  with TemporaryDirectory() as folder:
    path = Path(folder) / "shared_body.xml"
    path.write_text(xml)
    robot = load_robot(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for name, position in (("hinge", 0.8), ("slide", 0.07), ("hinge", -0.4)):
      robot.set_joint_position(name, position)
      data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
      check_poses(robot, model, data)


def main():
  check_joint_motions()
  check_initial_states()
  path = Path(__file__).resolve().parents[1] / "data/mjcf/franka_fr3/fr3.xml"
  robot = load_robot(path)
  model = mujoco.MjModel.from_xml_path(str(path))
  data = mujoco.MjData(model)
  source_vertices = [
    (geometry, geometry.vertices.copy())
    for part in robot.parts.values()
    for geometry in part.visuals + part.collisions
  ]
  viewer = Viewer(off_screen=True)
  viewer.add_robot(robot)
  try:
    with patch.object(viewer.plotter, "show"):
      viewer.show(show_collisions=False)
    viewer.plotter.show(auto_close=False)
    names = [joint.name for joint in robot.joints.values() if joint.dof == 1]
    widgets = dict(zip(names, viewer.plotter.widgets.slider_widgets, strict=True))
    assert len(widgets) == 9
    check_poses(robot, model, data)
    check_meshes(viewer, model, data)
    camera = np.asarray(viewer.plotter.camera_position).copy()
    for name, value in (
      ("fr3_joint1", 35),
      ("fr3_joint2", -30),
      ("fr3_joint4", -120),
      ("fr3_joint6", 110),
      ("fr3_finger_joint1", 0.01),
      ("fr3_finger_joint2", 0.025),
      ("fr3_joint1", -20),
      ("fr3_joint4", -90),
    ):
      widget = widgets[name]
      widget.GetRepresentation().SetValue(value)
      widget.InvokeEvent("InteractionEvent")
      joint_id = model.joint(name).id
      position = (
        value if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else np.deg2rad(value)
      )
      data.qpos[model.jnt_qposadr[joint_id]] = position
      np.testing.assert_allclose(robot.joints[name].position, position, atol=1e-12)
      check_poses(robot, model, data)
      check_meshes(viewer, model, data)
      np.testing.assert_array_equal(np.asarray(viewer.plotter.camera_position), camera)
    for geometry, original in source_vertices:
      np.testing.assert_array_equal(geometry.vertices, original)
    button = viewer.plotter.widgets.button_widgets[0]
    button.GetRepresentation().SetState(1)
    button.InvokeEvent("StateChangedEvent")
    assert viewer.plotter.actors["collision_0"].visibility
    check_meshes(viewer, model, data)
  finally:
    viewer.plotter.close()
  check_shared_body()
  print(
    "Passed: initial joint states, direct motions, nine FR3 sliders, child/descendant poses, "
    "meshes, axes, labels, and shared-body joints."
  )


if __name__ == "__main__":
  main()
