"""Run from the repository after `pip install -e '.[all]'`.

The bundled assets exercise real robot descriptions. Small temporary files
cover joint types, transformations, and invalid inputs absent from those assets.
No viewer, display, network access, or test framework is needed to run this.
"""

from math import pi
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
import unittest

from yarss import Loader, load_robot
from yarss.joints import (
  ContinuousJoint,
  D6Joint,
  DistanceJoint,
  FixedJoint,
  FloatingJoint,
  Joint,
  PlanarJoint,
  PrismaticJoint,
  RevoluteJoint,
  SphericalJoint,
)
from yarss.transforms import multiply, rpy_quaternion, transform

EXAMPLES = Path(__file__).resolve().parents[1] / "example" / "data"


class LoaderTests(unittest.TestCase):
  def setUp(self):
    self.directory = TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.folder = Path(self.directory.name)

  def write(self, name, text):
    path = self.folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path

  def test_bundled_examples(self):
    cases = [
      ("urdf/franka_description/urdf/fr3.urdf", 26, 25),
      ("mjcf/franka_fr3/fr3.xml", 12, 12),
      ("usd/cartpole.usda", 4, 4),
    ]
    robots = {}
    for filename, parts, joints in cases:
      with self.subTest(format=filename):
        robot = load_robot(EXAMPLES / filename)
        robots[robot.metadata["format"]] = robot
        self.assertEqual(len(robot.parts), parts)
        self.assertEqual(len(robot.joints), joints)
        self.assertEqual(len(robot.roots), 1)
        self.assertTrue(all(isinstance(j, Joint) for j in robot.joints.values()))
        # Real descriptions also contain geometry-free base, sensor, and tool frames.
        self.assertTrue(any(p.visuals for p in robot.parts.values()))
        for part in robot.parts.values():
          for shape in part.visuals + part.collisions:
            if shape.kind == "mesh":
              self.assertTrue(shape.vertices)
              self.assertTrue(shape.faces)
        print(
          f"\n  {robot.metadata['format'].upper()}: {robot.name}: {parts} parts, {joints} joints"
        )

    # Both sources describe the same arm. MJCF's reference geometry is the home
    # pose, while URDF keeps the manufacturer's zero-coordinate geometry.
    urdf, mjcf = robots["urdf"], robots["mjcf"]
    for robot in (urdf, mjcf):
      self.assertEqual(sum(isinstance(j, RevoluteJoint) for j in robot.joints.values()), 7)
      for i in range(8):
        part = robot.parts[f"fr3_link{i}"]
        self.assertTrue(part.visuals)
        self.assertTrue(part.collisions)
        self.assertTrue(all(s.kind == "mesh" for s in part.visuals + part.collisions))
    for i in range(1, 8):
      name = f"fr3_joint{i}"
      a, b = urdf.joints[name], mjcf.joints[name]
      for joint in (a, b):
        self.assertIsInstance(joint, RevoluteJoint)
        self.assertEqual((joint.parent, joint.child), (f"fr3_link{i - 1}", f"fr3_link{i}"))
        self.assertEqual(joint.axis, (0, 0, 1))
        limit = joint.limits["position"]
        self.assertLess(limit.lower, limit.upper)
      reference = b.metadata["reference_position"][0]
      home_rotation = transform(quaternion=rpy_quaternion((0, 0, reference)))
      for frame_a, frame_b in (
        (multiply(a.parent_frame, home_rotation), b.parent_frame),
        (a.child_frame, b.child_frame),
      ):
        for row_a, row_b in zip(frame_a, frame_b):
          for value_a, value_b in zip(row_a, row_b):
            self.assertAlmostEqual(value_a, value_b)
    for robot in (urdf, mjcf):
      self.assertEqual(sum(isinstance(j, PrismaticJoint) for j in robot.joints.values()), 2)
      for i in (1, 2):
        finger = robot.joints[f"fr3_finger_joint{i}"]
        self.assertIsInstance(finger, PrismaticJoint)
        self.assertEqual(finger.parent, "fr3_hand")
        self.assertEqual(finger.axis, (0, 1, 0))
        self.assertEqual(finger.limits["position"].lower, 0)
        self.assertEqual(finger.limits["position"].upper, 0.04)
        self.assertTrue(robot.parts[finger.child].visuals)
        self.assertTrue(robot.parts[finger.child].collisions)
    self.assertEqual(urdf.joints["fr3_finger_joint2"].mimic.joint, "fr3_finger_joint1")
    self.assertEqual(mjcf.metadata["actuator_count"], 8)
    self.assertEqual(mjcf.metadata["tendon_count"], 1)
    self.assertEqual(mjcf.metadata["equality_constraint_count"], 1)
    # Compare the two descriptions at the same joint coordinates, not their
    # different initial poses. The wrist and fingers must still match the URDF.
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf.source))
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_kinematics(model, data)
    names = [(f"fr3_link{i}", f"fr3_link{i}") for i in range(8)] + [
      ("fr3_hand", "fr3_hand"),
      ("fr3_leftfinger", "fr3_left_finger"),
      ("fr3_rightfinger", "fr3_right_finger"),
    ]
    for urdf_name, mjcf_name in names:
      body = model.body(mjcf_name).id
      zero_pose = transform(data.xpos[body], data.xquat[body])
      for row_a, row_b in zip(urdf.parts[urdf_name].transform, zero_pose):
        for value_a, value_b in zip(row_a, row_b):
          self.assertAlmostEqual(value_a, value_b)

  def test_mjcf_fr3_gripper_opens_and_closes(self):
    import mujoco
    import numpy as np

    # Copy only the FR3 folder: the entire robot must work without sibling packages.
    folder = copytree(EXAMPLES / "mjcf/franka_fr3", self.folder / "franka_fr3")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    self.assertEqual(data.ncon, 0, "The initial robot must not intersect itself or the floor")
    np.testing.assert_allclose(data.qpos, model.key("home").qpos)
    self.assertTrue(np.all(data.qpos >= model.jnt_range[:, 0]))
    self.assertTrue(np.all(data.qpos <= model.jnt_range[:, 1]))
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    fingers = [model.joint(f"fr3_finger_joint{i}").qposadr[0] for i in (1, 2)]
    actuator = model.actuator("fr3_actuator8").id
    self.assertEqual((model.nq, model.nu), (9, 8))
    for address in fingers:
      self.assertAlmostEqual(data.qpos[address], 0.04)
    hand = model.body("fr3_hand").id
    finger_bodies = [model.body(f"fr3_{side}_finger").id for side in ("left", "right")]
    mujoco.mj_forward(model, data)
    rotations = data.xmat[finger_bodies].copy()
    for opening in (0, 0.02, 0.04):
      data.qpos[fingers] = opening
      mujoco.mj_forward(model, data)
      # A prismatic joint translates: neither finger's orientation may change.
      np.testing.assert_allclose(data.xmat[finger_bodies], rotations, atol=1e-12)
      local = (data.xpos[finger_bodies] - data.xpos[hand]) @ data.xmat[hand].reshape(3, 3)
      np.testing.assert_allclose(local, [[0, opening, 0.0584], [0, -opening, 0.0584]], atol=1e-12)
      # Closed pads may just touch; allow numerical contact at that endpoint.
      for contact in data.contact:
        bodies = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        self.assertEqual(bodies, set(finger_bodies))
        self.assertGreaterEqual(contact.dist, -1e-9)
    for command, expected in ((0, 0), (255, 0.04)):
      data.ctrl[actuator] = command
      for _ in range(1500):
        mujoco.mj_step(model, data)
      for address in fingers:
        self.assertAlmostEqual(data.qpos[address], expected, delta=0.002)
      self.assertAlmostEqual(data.qpos[fingers[0]], data.qpos[fingers[1]], delta=0.0001)
    self.assertFalse(data.warning.number.any())

  def test_urdf_joint_types_and_mimic(self):
    kinds = {
      "fixed": FixedJoint,
      "revolute": RevoluteJoint,
      "continuous": ContinuousJoint,
      "prismatic": PrismaticJoint,
      "planar": PlanarJoint,
      "floating": FloatingJoint,
    }
    links = '<link name="base"/>'
    joints = ""
    parent = "base"
    for kind in kinds:
      links += f'<link name="{kind}"/>'
      joints += f'''<joint name="{kind}" type="{kind}">
                <parent link="{parent}"/><child link="{kind}"/>
                <axis xyz="0 0 2"/><limit lower="-1" upper="2" effort="3" velocity="4"/>
                </joint>'''
      parent = kind
    robot = load_robot(self.write("types.urdf", f'<robot name="types">{links}{joints}</robot>'))
    for kind, cls in kinds.items():
      self.assertIs(type(robot.joints[kind]), cls)
      self.assertEqual(robot.joints[kind].axis, (0, 0, 1))
    self.assertIsNone(robot.joints["continuous"].limits["position"].lower)
    mimic = """<joint name="mimic" type="revolute"><parent link="base"/>
            <child link="follower"/><mimic joint="revolute" multiplier="-2" offset="0.5"/>
            <limit lower="-1" upper="2" effort="3" velocity="4"/></joint>"""
    robot = load_robot(
      self.write("mimic.urdf", f'<robot>{links}<link name="follower"/>{joints}{mimic}</robot>')
    )
    self.assertEqual(robot.joints["mimic"].mimic.multiplier, -2)

  def test_urdf_mesh_package_paths_and_transforms(self):
    mesh = self.write("somewhere/triangle.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    path = self.write(
      "robot.urdf",
      f"""<robot name="transforms">
          <link name="base"/><link name="arm"><visual><origin xyz="1 0 0"/>
            <geometry><mesh filename="package://test/triangle.obj" scale="2 3 4"/></geometry>
          </visual></link><link name="tip"/>
          <joint name="rotate" type="fixed"><parent link="base"/><child link="arm"/>
            <origin xyz="1 2 3" rpy="0 0 {pi / 2}"/></joint>
          <joint name="offset" type="fixed"><parent link="arm"/><child link="tip"/>
            <origin xyz="1 0 0"/></joint></robot>""",
    )
    robot = Loader(package_paths={"test": mesh.parent}).load(path)
    shape = robot.parts["arm"].visuals[0]
    self.assertEqual(shape.mesh_path, mesh)
    self.assertEqual(len(shape.faces), 1)
    self.assertEqual([shape.transform[i][i] for i in range(3)], [2, 3, 4])
    tip = robot.parts["tip"].transform
    for value, expected in zip((tip[i][3] for i in range(3)), (1, 3, 3)):
      self.assertAlmostEqual(value, expected)
    with self.assertRaisesRegex(FileNotFoundError, "package_paths"):
      load_robot(path)

  def test_invalid_files_and_urdf_topology(self):
    invalid = [
      "<robot/>",
      '<robot><link name="a"/><link name="a"/></robot>',
      '<robot><link name="a"/><link name="b"/></robot>',
      '<robot><link name="a"/><joint name="j" type="unknown"/></robot>',
      '<robot><link name="a"/><joint name="j" type="fixed"><parent link="a"/><child link="missing"/></joint></robot>',
      '<robot><link name="a"/><link name="b"/><joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="ba" type="fixed"><parent link="b"/><child link="a"/></joint></robot>',
    ]
    for xml in invalid:
      with self.subTest(xml=xml), self.assertRaises(ValueError):
        load_robot(self.write("invalid.urdf", xml))
    with self.assertRaises(FileNotFoundError):
      load_robot(self.folder / "missing.urdf")
    with self.assertRaisesRegex(ValueError, "Unsupported"):
      load_robot(self.write("robot.txt", "robot"))
    with self.assertRaisesRegex(ValueError, "Expected"):
      load_robot(self.write("other.xml", "<scene/>"))

  def test_mjcf_defaults_includes_joint_types_and_fixed_parts(self):
    self.write(
      "bodies.xml",
      """<mujocoinclude>
          <body name="base"><geom type="sphere" size=".1"/>
            <body name="slider" pos="0 0 1"><joint name="slide" type="slide" range="-2 3"/>
              <joint name="hinge" type="hinge" range="-90 90"/><geom type="sphere" size=".1"/></body>
            <body name="ball"><joint name="ball" type="ball" range="0 90"/>
              <geom type="sphere" size=".1"/></body>
            <body name="wheel"><joint name="spin" limited="false"/>
              <geom type="sphere" size=".1"/></body>
          </body>
          <body name="free"><freejoint name="free"/><geom type="sphere" size=".1"/></body>
        </mujocoinclude>""",
    )
    path = self.write(
      "robot.mjcf",
      """<mujoco model="types">
          <compiler angle="degree" fusestatic="true" discardvisual="true"/>
          <default><joint type="hinge" limited="true"/><geom mass="1"/></default>
          <worldbody><include file="bodies.xml"/></worldbody></mujoco>""",
    )
    robot = load_robot(path)
    self.assertEqual(len(robot.parts), 5)
    self.assertEqual(len(robot.joints), 6)
    self.assertIsInstance(robot.joints["slide"], PrismaticJoint)
    self.assertIsInstance(robot.joints["ball"], SphericalJoint)
    self.assertIsInstance(robot.joints["free"], FloatingJoint)
    self.assertIsInstance(robot.joints["spin"], ContinuousJoint)
    self.assertAlmostEqual(robot.joints["hinge"].limits["position"].lower, -pi / 2)
    self.assertEqual(robot.joints["slide"].limits["position"].lower, -2)
    self.assertEqual(robot.joints["hinge"].parent, "base")
    self.assertEqual(robot.joints["hinge"].child, "slider")
    self.assertEqual(robot.joints["hinge"].metadata["order_in_body"], 1)
    self.assertEqual(robot.joints["hinge"].parent_frame[2][3], 1)

  def test_mjcf_generated_names_and_explicit_contact_pairs(self):
    path = self.write(
      "names.xml",
      """<mujoco>
          <default><geom type="sphere" size=".1" contype="0" conaffinity="0"/></default>
          <worldbody>
            <body><geom name="a"/></body>
            <body name="__body_1"><joint name="__fixed_1"/><geom name="b"/></body>
          </worldbody>
          <contact><pair geom1="a" geom2="b"/></contact></mujoco>""",
    )
    robot = load_robot(path)
    self.assertEqual(len(robot.parts), 2)
    self.assertEqual(len(robot.joints), 2)
    self.assertIn("__body_1", robot.parts)
    self.assertIsInstance(robot.joints["__fixed_1"], ContinuousJoint)
    self.assertTrue(all(p.collisions for p in robot.parts.values()))

  def test_mjcf_inline_mesh_scale_is_applied_once(self):
    path = self.write(
      "mesh.xml",
      """<mujoco><asset>
          <mesh name="tetra" scale="2 3 4" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/></asset>
          <worldbody><body name="mesh" pos="1 2 3"><geom type="mesh" mesh="tetra"/></body></worldbody>
          </mujoco>""",
    )
    robot = load_robot(path)
    part = robot.parts["mesh"]
    shape = part.visuals[0]
    self.assertEqual((len(shape.vertices), len(shape.faces)), (4, 4))
    world = multiply(part.transform, shape.transform)
    points = [
      tuple(sum(world[i][j] * p[j] for j in range(3)) + world[i][3] for i in range(3))
      for p in shape.vertices
    ]
    for axis, extent in enumerate((2, 3, 4)):
      self.assertAlmostEqual(
        max(p[axis] for p in points) - min(p[axis] for p in points), extent, places=5
      )

  def test_usd_schemas_units_meshes_and_binary_formats(self):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    root.AddTranslateOp().Set((100, 0, 0))
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in ("a", "b"):
      body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
      body.AddTranslateOp().Set((0, 200, 0))
      UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Robot/a/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    specs = [
      ("fixed", UsdPhysics.FixedJoint, FixedJoint),
      ("hinge", UsdPhysics.RevoluteJoint, RevoluteJoint),
      ("slide", UsdPhysics.PrismaticJoint, PrismaticJoint),
      ("ball", UsdPhysics.SphericalJoint, SphericalJoint),
      ("distance", UsdPhysics.DistanceJoint, DistanceJoint),
      ("d6", UsdPhysics.Joint, D6Joint),
    ]
    for name, schema, _ in specs:
      joint = schema.Define(stage, f"/Robot/{name}")
      joint.CreateBody0Rel().SetTargets(["/Robot/a"])
      joint.CreateBody1Rel().SetTargets(["/Robot/b"])
      joint.CreateLocalPos0Attr(Gf.Vec3f(100, 0, 0))
      if name in {"hinge", "slide"}:
        joint.CreateLowerLimitAttr(-90)
        joint.CreateUpperLimitAttr(90)
      if name == "ball":
        joint.CreateConeAngle0LimitAttr(45)
      if name == "distance":
        joint.CreateMinDistanceAttr(10)
        joint.CreateMaxDistanceAttr(20)
      if name == "d6":
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), "rotX")
        limit.CreateLowAttr(1)
        limit.CreateHighAttr(-1)  # Locked axis, not a malformed range.
    for suffix in (".usda", ".usd", ".usdc"):
      path = self.folder / ("robot" + suffix)
      stage.GetRootLayer().Export(str(path))
      robot = load_robot(path, usd_root="/Robot")
      for name, _, kind in specs:
        self.assertIs(type(robot.joints[f"/Robot/{name}"]), kind)
      self.assertEqual(robot.parts["/Robot/a"].transform[0][3], 1)
      self.assertEqual(robot.parts["/Robot/a"].transform[1][3], 2)
      shape = robot.parts["/Robot/a"].visuals[0]
      self.assertEqual(shape.vertices[1], (1, 0, 0))
      self.assertEqual(shape.faces, ((0, 1, 2),))
      self.assertEqual(robot.metadata["up_axis"], "Y")
      self.assertEqual(robot.joints["/Robot/fixed"].parent_frame[0][3], 1)
      self.assertAlmostEqual(robot.joints["/Robot/hinge"].limits["position"].upper, pi / 2)
      self.assertAlmostEqual(robot.joints["/Robot/slide"].limits["position"].upper, 0.9)
      self.assertAlmostEqual(robot.joints["/Robot/ball"].limits["coneAngle0Limit"].upper, pi / 4)
      self.assertAlmostEqual(robot.joints["/Robot/distance"].limits["position"].upper, 0.2)
      self.assertGreater(
        robot.joints["/Robot/d6"].limits["rotX"].lower,
        robot.joints["/Robot/d6"].limits["rotX"].upper,
      )

  def test_usd_references_instances_and_world_endpoint(self):
    from pxr import Usd, UsdGeom, UsdPhysics

    source = Usd.Stage.CreateNew(str(self.folder / "source.usda"))
    UsdGeom.SetStageMetersPerUnit(source, 1)
    body = UsdGeom.Xform.Define(source, "/body")
    source.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdGeom.Cube.Define(source, "/body/geometry")
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(self.folder / "scene.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    part = UsdGeom.Xform.Define(stage, "/Robot/instance").GetPrim()
    part.GetReferences().AddReference("source.usda")
    part.SetInstanceable(True)
    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/fixed")
    joint.CreateBody0Rel().SetTargets(["/Robot/instance"])
    stage.GetRootLayer().Save()
    robot = load_robot(self.folder / "scene.usda")
    self.assertEqual(len(robot.parts["/Robot/instance"].visuals), 1)
    self.assertIsNone(robot.joints["/Robot/fixed"].child)
    self.assertEqual(len(robot.connections("/Robot/instance")), 1)
    joint.CreateBody1Rel().SetTargets(["/missing"])
    stage.GetRootLayer().Save()
    with self.assertRaisesRegex(ValueError, "missing USD prim"):
      load_robot(self.folder / "scene.usda")


if __name__ == "__main__":
  unittest.main(verbosity=2)
