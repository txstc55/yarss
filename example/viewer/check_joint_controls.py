"""Run from the repository root: python3 example/viewer/check_joint_controls.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np
import pyvista as pv

from yarss import Part, Robot, load_robot
from yarss.joints import FixedJoint, RotatingJoint, SlidingJoint, UnlimitedRotatingJoint
from yarss.transforms import rotation_transform
from yarss.viewer import Viewer


def check_joint_motions():
  """Check a diagonal, non-unit axis with known movements and nonzero initial values."""
  parent, child = Part("parent"), Part("child")
  axis = np.array([0.0, 1.0, 1.0])
  diagonal = 1 / np.sqrt(2)
  for kind in (RotatingJoint, UnlimitedRotatingJoint):
    joint = kind("turn", parent, child, axis=axis, initial_value=30.0)
    joint.value = 30.0
    np.testing.assert_allclose(joint.motion_transform(), np.eye(4), atol=1e-12)
    for angle in (90.0, 810.0):
      joint.value = 30.0 + angle
      motion = joint.motion_transform()
      np.testing.assert_allclose(motion[:3, 0], [0, diagonal, -diagonal], atol=1e-12)
      np.testing.assert_allclose(motion[:3, :3] @ axis, axis, atol=1e-12)
      np.testing.assert_allclose(motion.T @ motion, np.eye(4), atol=1e-12)
      np.testing.assert_allclose(np.linalg.det(motion[:3, :3]), 1, atol=1e-12)

  joint = SlidingJoint(
    "slide",
    parent,
    child,
    axis=axis,
    value=0.14,
    initial_value=0.04,
  )
  expected = np.eye(4)
  expected[:3, 3] = [0, 0.1 * diagonal, 0.1 * diagonal]
  np.testing.assert_allclose(joint.motion_transform(), expected, atol=1e-12)
  np.testing.assert_array_equal(FixedJoint("fixed", parent, child).motion_transform(), np.eye(4))
  np.testing.assert_array_equal(axis, [0, 1, 1])


def check_joint_parts_and_inverse():
  """Use shared Part objects and invert a rotated, offset child frame only once."""
  parent, child = Part("parent"), Part("child")
  anchor = rotation_transform(np.array([1.0, 2.0, 3.0]), 35.0)
  anchor[:3, 3] = [0.1, 0.2, 0.3]
  with patch("numpy.linalg.inv", wraps=np.linalg.inv) as invert:
    joint = RotatingJoint("turn", parent, child, parent_frame=anchor.copy(), child_frame=anchor)
    assert invert.call_count == 1
  inverse = joint.child_frame_inverse
  np.testing.assert_allclose(joint.child_frame @ inverse, np.eye(4), atol=1e-12)
  assert not joint.child_frame.flags.writeable
  assert not inverse.flags.writeable
  anchor[0, 3] = 100.0
  assert joint.child_frame[0, 3] == 0.1

  robot = Robot("parts", Path(__file__), metadata={"format": "mjcf"})
  robot.add_part(parent)
  robot.add_part(child)
  robot.add_joint(FixedJoint("world", None, parent))
  robot.add_joint(joint)
  robot.finish_setup()
  assert joint.parent is robot.parts["parent"]
  assert joint.child is robot.parts["child"]
  assert robot.roots == [parent]
  assert [j.name for j in robot.connections("parent")] == ["world", "turn"]
  assert [j.name for j in robot.connections("child")] == ["turn"]
  with patch("numpy.linalg.inv", side_effect=AssertionError("Use the cached inverse")):
    for value in (45.0, -30.0, 0.0):
      robot.set_joint_value(joint, value)
      robot.joint_world_frame("turn")
  assert joint.child_frame_inverse is inverse
  np.testing.assert_allclose(child.transform, np.eye(4), atol=1e-12)

  # Matching names are insufficient: joints must point at the registered Part itself.
  joint.child = Part("child")
  try:
    robot.validate()
  except ValueError as error:
    assert "unregistered part 'child'" in str(error)
  else:
    raise AssertionError("An unregistered Part with the same name must be rejected")


def check_initial_states():
  """MJCF files in either angle unit load joint angles and limits as degrees."""
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
      <joint name="distance" type="slide" ref="0.04" range="0 0.08"/>
      <geom size="0.1"/>
    </body>
    <body name="wrist" euler="0.2 0.3 0.4">
      <joint name="orientation" type="ball" range="0 0.8"/>
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
    for unit in ("radian", "degree"):
      path.write_text(xml.replace('angle="radian"', f'angle="{unit}"'))
      robot = load_robot(path)
      scale = 180 / np.pi if unit == "radian" else 1.0
      np.testing.assert_allclose(robot.joints["angle"].initial_value, 0.3 * scale)
      np.testing.assert_allclose(robot.joints["spin"].initial_value, 0.7 * scale)
      limit = robot.joints["angle"].limits["value"]
      np.testing.assert_allclose([limit.lower, limit.upper], [-scale, scale])
      np.testing.assert_allclose(robot.joints["orientation"].limits["angle"].upper, 0.8 * scale)
      assert robot.joints["distance"].initial_value == 0.04
      limit = robot.joints["distance"].limits["value"]
      np.testing.assert_allclose([limit.lower, limit.upper], [0.0, 0.08])
      np.testing.assert_array_equal(robot.joints["orientation"].initial_value, np.eye(3))
      np.testing.assert_allclose(
        robot.joints["pose"].initial_value, robot.parts["free_body"].transform, atol=1e-12
      )
      for joint in robot.joints.values():
        assert joint.parent is None
        assert joint.child is robot.parts[joint.child.name]
        np.testing.assert_allclose(
          joint.child_frame @ joint.child_frame_inverse, np.eye(4), atol=1e-12
        )
        np.testing.assert_array_equal(joint.value, joint.initial_value)
        if isinstance(joint.value, np.ndarray):
          assert not np.shares_memory(joint.value, joint.initial_value)
      for name, value in (("angle", 0.6), ("spin", 2.0), ("distance", 0.02)):
        joint = robot.joints[name]
        initial = joint.initial_value
        assert joint.value == initial
        robot.set_joint_value(joint, value)
        assert joint.initial_value == initial
      for joint in robot.joints.values():
        assert set(joint.metadata) == {"order_in_body", "stiffness"}


def check_finish_setup():
  """Cache a branching tree independently of part insertion order, and rebuild after edits."""
  base, arm, left, tip, right, other = [
    Part(name) for name in ("base", "arm", "left", "tip", "right", "other")
  ]
  hinge = RotatingJoint("hinge", base, arm)
  robot = Robot("setup", Path(__file__), metadata={"format": "mjcf"})
  for part in (tip, other, right, arm, left, base):
    robot.add_part(part)
  for joint in (
    FixedJoint("root", None, base),
    hinge,
    FixedJoint("left", arm, left),
    FixedJoint("tip", left, tip),
    FixedJoint("right", arm, right),
    FixedJoint("other", None, other),
  ):
    robot.add_joint(joint)
  robot.finish_setup()
  assert base.affected_parts == [base, arm, left, tip, right]
  assert arm.affected_parts == [arm, left, tip, right]
  assert left.affected_parts == [left, tip]
  assert tip.affected_parts == [tip]
  assert other.affected_parts == [other]
  cached = arm.affected_parts
  other_pose = other.transform
  for value in (30.0, -45.0, 0.0):
    assert robot.set_joint_value(hinge, value) is cached
    assert other.transform is other_pose

  # Non-finite values must fail before changing the joint state or any part pose.
  arm_pose = arm.transform
  for value in (np.nan, np.inf, -np.inf):
    try:
      robot.set_joint_value(hinge, value)
    except ValueError:
      pass
    else:
      raise AssertionError("Invalid scalar updates must be rejected")
    assert arm.transform is arm_pose
    assert hinge.value == 0.0

  extra = Part("extra")
  robot.add_part(extra)
  robot.add_joint(FixedJoint("extra", right, extra))
  robot.finish_setup()
  assert arm.affected_parts == [arm, left, tip, right, extra]
  assert robot.set_joint_value(hinge, 10.0) is arm.affected_parts

  # A broken graph must fail during setup rather than loop forever.
  tip.add_connected_child(base)
  try:
    robot.finish_setup()
  except ValueError as error:
    assert "requires a tree" in str(error)
  else:
    raise AssertionError("A cycle must be rejected during setup")


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
    body_id = model.jnt_bodyid[joint_id]
    parent_id = model.body_parentid[body_id]
    assert joint.child is robot.parts[model.body(body_id).name]
    assert joint.parent is (robot.parts[model.body(parent_id).name] if parent_id else None)
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
      actor = viewer.plotter.actors[f"{prefix}_{index}"]
      matrix = actor.user_matrix
      actor_mesh = actor.mapper.dataset
      displayed_points = actor_mesh.points @ matrix[:3, :3].T + matrix[:3, 3]
      if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start : start + count].astype(float)
        world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        np.testing.assert_allclose(displayed_points, world, atol=1e-7)
      else:
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        center = np.asarray(actor_mesh.center) @ matrix[:3, :3].T + matrix[:3, 3]
        np.testing.assert_allclose(center, data.geom_xpos[geom_id], atol=1e-7)
      np.testing.assert_array_equal(actor_mesh.points, mesh.points)

  expected_labels = []
  for _, joint, actor, length in viewer._joint_axes:
    joint_id = model.joint(joint.name).id
    expected_labels.append(data.xanchor[joint_id] + length * data.xaxis[joint_id])
    frame = actor.user_matrix
    np.testing.assert_allclose(frame[:3, 3], data.xanchor[joint_id], atol=1e-12)
    np.testing.assert_allclose(frame[:3, :3] @ joint.axis, data.xaxis[joint_id], atol=1e-12)
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
    base, moving, tip = (robot.parts[name] for name in ("base", "moving", "tip"))
    assert base.connected_parents == [None]
    assert base.connected_children == [moving]
    assert moving.connected_parents == [base]
    assert moving.connected_children == [tip]
    assert [joint.name for joint in moving.incoming_joints] == ["hinge", "slide"]
    hinge = robot.joints["hinge"]
    assert robot.set_joint_value(hinge, hinge.value) == [moving, tip]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    viewer = Viewer(off_screen=True)
    viewer.add_robot(robot)
    try:
      with patch.object(viewer.plotter, "show"):
        viewer.show()
      widgets = dict(zip(("hinge", "slide"), viewer.plotter.widgets.slider_widgets, strict=True))
      for name, span, initial in (("hinge", 180.0, np.rad2deg(0.2)), ("slide", 0.1, 0.03)):
        representation = widgets[name].GetRepresentation()
        np.testing.assert_allclose(
          [representation.GetMinimumValue(), representation.GetMaximumValue()],
          [initial - span, initial + span],
        )
      for name, value in (("hinge", 45.0), ("slide", 0.07), ("hinge", -25.0)):
        widgets[name].GetRepresentation().SetValue(value)
        widgets[name].InvokeEvent("InteractionEvent")
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = (
          value if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else np.deg2rad(value)
        )
        check_poses(robot, model, data)
    finally:
      viewer.plotter.close()


def check_viewer():
  """Actor transforms must match MuJoCo while all robot and arrow meshes stay static."""
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
    for name, widget in widgets.items():
      joint = robot.joints[name]
      joint_id = model.joint(name).id
      linear = model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE
      scale = 1.0 if linear else 180 / np.pi
      initial = model.qpos0[model.jnt_qposadr[joint_id]] * scale
      lower, upper = model.jnt_range[joint_id] * scale
      np.testing.assert_allclose(joint.value, initial, atol=1e-12)
      representation = widget.GetRepresentation()
      np.testing.assert_allclose(
        [
          representation.GetValue(),
          representation.GetMinimumValue(),
          representation.GetMaximumValue(),
        ],
        [initial, lower, upper],
        atol=1e-12,
      )
    check_poses(robot, model, data)
    check_meshes(viewer, model, data)
    camera = np.asarray(viewer.plotter.camera_position).copy()
    actors = [actor for group in viewer._part_actors.values() for actor in group]
    actors.extend(actor for _, _, actor, _ in viewer._joint_axes)
    static_meshes = [(actor.mapper.dataset, actor.mapper.dataset.GetMTime()) for actor in actors]
    movements = [
      ("fr3_joint1", 35),
      ("fr3_joint2", -30),
      ("fr3_joint4", -120),
      ("fr3_joint6", 110),
      ("fr3_finger_joint1", 0.01),
      ("fr3_finger_joint2", 0.025),
      ("fr3_joint1", -20),
      ("fr3_joint4", -90),
    ]
    for name in names:
      joint_id = model.joint(name).id
      scale = 1.0 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else 180 / np.pi
      lower, upper = model.jnt_range[joint_id] * scale
      initial = model.qpos0[model.jnt_qposadr[joint_id]] * scale
      movements.extend((name, value) for value in (lower, upper, (lower + upper) / 2, initial))
    for name, value in movements:
      previous_poses = {part: part.transform for part in robot.parts.values()}
      actor_times = [
        (part, actor, actor.GetMTime())
        for part, group in viewer._part_actors.items()
        for actor in group
      ]
      axis_times = [actor.GetMTime() for _, _, actor, _ in viewer._joint_axes]
      widget = widgets[name]
      widget.GetRepresentation().SetValue(value)
      with patch.object(
        pv.PolyData, "compute_normals", side_effect=AssertionError("Meshes must stay static")
      ):
        widget.InvokeEvent("InteractionEvent")
      for mesh, before in static_meshes:
        assert mesh.GetMTime() == before
      for part, actor, before in actor_times:
        if part.transform is previous_poses[part]:
          assert actor.GetMTime() == before
      for (_, joint, actor, _), before in zip(viewer._joint_axes, axis_times, strict=True):
        if joint.child.transform is previous_poses[joint.child]:
          assert actor.GetMTime() == before
      joint_id = model.joint(name).id
      qpos = value if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else np.deg2rad(value)
      data.qpos[model.jnt_qposadr[joint_id]] = qpos
      np.testing.assert_allclose(robot.joints[name].value, value, atol=1e-12)
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


def main():
  check_joint_motions()
  check_joint_parts_and_inverse()
  check_initial_states()
  check_finish_setup()
  check_viewer()
  check_shared_body()
  print(
    "Passed: setup order and rebuilds, finite-value checks, shared Part endpoints, "
    "cached child-frame inverses, degree-based states and limits, "
    "direct motions, all nine FR3 sliders at both limits, "
    "midpoints, and initial positions, child/descendant poses, "
    "meshes, axes, labels, and unbounded sliders with shared-body joints."
  )


if __name__ == "__main__":
  main()
