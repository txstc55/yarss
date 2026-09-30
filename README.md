# YARSS — Yet Another Robotics Simulation System

This first version loads robot descriptions into a shared `Robot` containing
`Part` objects and typed `Joint` connections. It extracts geometry, default
poses, joint frames, axes, and limits. Joint motion and physics are placeholders.

## Setup and test

From this repository:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[all]'
.venv/bin/python scripts/test_loaders.py
```

Python 3.10+ is required. URDF uses Python's XML reader, trimesh for mesh files,
and pycollada for COLLADA geometry. The `mjcf` and `usd` extras install MuJoCo
and OpenUSD respectively; `all` installs both. Neither a GUI nor a GPU is needed.
The examples are included, so the checks run offline after installation.

The test script checks real assets and temporary descriptions covering joint
types, units, transforms, mesh data, includes/references, and invalid inputs.
It also checks the gripper's wrist attachment across formats and opens and closes
both fingers using native MuJoCo without a viewer.

| Bundled example | Parts | Joints |
| --- | ---: | ---: |
| Franka Research 3 with parallel gripper, manufacturer URDF | 26 | 25 |
| Franka Research 3 with parallel gripper, Menagerie-based MJCF | 12 | 12 |
| Newton cart-pole, USD | 4 | 4 |

Both FR3 examples contain seven revolute arm joints and the Franka Hand's two
prismatic finger joints. Each finger slides from 0 to 40 mm; the fingers move
symmetrically for an opening of up to 80 mm. Visual meshes and collision geometry
are included for the arm, hand, and fingers.

Each format provides a complete FR3 with its gripper in one robot package.
The URDF uses a mimic joint to couple the fingers. The MJCF's `fr3.xml` contains
the arm, hand, fingers, equality constraint, tendon, and gripper actuator directly;
all their meshes live together in `franka_fr3/assets/`. These coupling definitions
remain in the robot files;
YARSS's joint classes do not simulate them yet.

The URDF also contains a base frame, twelve accelerometer frames, a flange frame,
and a tool-center frame. The MJCF has three implicit fixed connections, including
its attachment to the world and the hand's attachment to the arm.
The sources specify different joint limits, which are preserved as supplied.
Asset sources and licenses are recorded in [example/data/README.md](example/data/README.md).

## Package layout

```text
YARSS/
├── pyproject.toml
├── yarss/
│   ├── loader/
│   │   ├── loader.py       # Loader and load_robot entry points
│   │   ├── urdf.py
│   │   ├── mjcf.py
│   │   └── usd.py
│   ├── joints/
│   │   ├── joint.py        # Base Joint, limits, and mimic definitions
│   │   └── ...            # One file per joint subclass
│   ├── parts/part.py      # Part and Geometry
│   ├── robot/robot.py     # Parts and their joint graph
│   └── transforms.py      # Static pose conversion helpers
├── example/data/           # Freely licensed robot and gripper assets
└── scripts/
    ├── test_loaders.py
    └── generate_fr3_urdf.py
```

## Loading a robot

```python
from yarss import load_robot

robot = load_robot("example/data/urdf/franka_description/urdf/fr3.urdf")

print(robot.name)
print([part.name for part in robot.roots])

for joint in robot.joints.values():
  print(joint.name, joint.kind, joint.parent, "->", joint.child)

wrist = robot.parts["fr3_link7"]
mesh = wrist.visuals[0]
print(mesh.mesh_path, len(mesh.vertices), len(mesh.faces))
print(robot.connections("fr3_link7"))
```

For the MuJoCo version, load `example/data/mjcf/franka_fr3/fr3.xml`.
The accompanying `scene.xml` includes the arm with its gripper, floor, lighting,
and viewing settings. Both formats name the finger joints `fr3_finger_joint1`
and `fr3_finger_joint2`.

`Loader().load(path)` is equivalent to `load_robot(path)`. Both accept paths
as strings or `pathlib.Path` objects. XML is identified by its root element:
`<robot>` is URDF, `<mujoco>` is MJCF. USD uses OpenUSD's native reader, including
binary layers and composed references.

For a ROS package stored elsewhere, provide its directory explicitly:

```python
from yarss import Loader

loader = Loader(package_paths={"my_robot_description": "/path/to/my_robot_description"})
robot = loader.load("/path/to/robot.urdf")
```

`package://` references also resolve automatically when the URDF is inside a
directory named after its ROS package, as in the bundled example. Relative and
local `file://` mesh paths are supported. Missing files raise an error.

For a USD scene containing several robots, select the subtree that contains
the desired bodies **and joints**:

```python
robot = load_robot("scene.usd", usd_root="/World/MyRobot")
```

Without `usd_root`, all rigid bodies and standard physics joints in the stage
are collected. USD part/joint names are full prim paths to avoid collisions.
Empty USD joint endpoints and MJCF world attachments are represented by `None`.

## Joint classes

Every type directly subclasses `Joint`; each file describes its intended
constraints in comments. There are no constraint solvers or motion methods yet.

| Class | Intended motion | Imported from |
| --- | --- | --- |
| `FixedJoint` | No relative motion | URDF, implicit MJCF, USD |
| `RevoluteJoint` | Limited rotation about one axis | URDF, MJCF hinge, USD |
| `ContinuousJoint` | Unlimited rotation about one axis | URDF, unlimited MJCF/USD hinge |
| `PrismaticJoint` | Translation along one axis | URDF, MJCF slide, USD |
| `PlanarJoint` | Two translations and one rotation in a plane | URDF |
| `FloatingJoint` | Three translations and three rotations | URDF, MJCF free |
| `SphericalJoint` | Rotation about a common point | MJCF ball, USD |
| `DistanceJoint` | Fixed or bounded anchor separation | USD |
| `D6Joint` | Individually free, limited, or locked axes | USD generic joint |
| `CylindricalJoint` | Independent slide and spin on one axis | Placeholder |
| `UniversalJoint` | Rotation about two intersecting axes | Placeholder |
| `ScrewJoint` | Rotation coupled to translation by pitch | Placeholder |

URDF mimic relationships are retained as metadata on the original joint type.
MJCF's multiple joints on a body share endpoints and retain their source order.
`Robot` stores a graph, so USD loops and parallel connections are allowed;
URDF descriptions are separately checked to be one connected tree.

## Geometry and coordinate conventions

- `Part.visuals` and `Part.collisions` contain `Geometry` objects. Geometry-free
  links are retained because joints can connect through them.
- Meshes contain `vertices` and `faces`. URDF meshes also retain `mesh_path`.
  USD polygons remain polygons; primitives remain shape/dimension records.
- `Part.transform` maps part coordinates to world coordinates at the file's
  default pose. `Geometry.transform` maps geometry coordinates to part coordinates.
  Joint anchor matrices map the joint frame into each endpoint's part frame.
- Matrices use column vectors. `part.transform @ geometry.transform @ vertex`
  gives the world vertex when using a matrix library. Mesh scale is already
  accounted for in either the geometry matrix or compiled vertices.
- Lengths are in meters and angles in radians. USD stage units are converted;
  its up axis is retained in `robot.metadata['up_axis']`. USD part matrices
  retain authored scale, including nonuniform scale.
- Scalar joint limits use `joint.limits['position']`. MJCF ball limits use
  `'angle'`; USD spherical joints use `'coneAngle0Limit'`/`'coneAngle1Limit'`.
  D6 limits use their native axis names, such as `'transX'` and `'rotZ'`.

## Current scope

This is a description loader, not a simulation engine. It does not calculate
joint motion, forces, motors, contacts, or enforce mimic relationships. Material
rendering and texture data are not imported into `Geometry`.

MJCF is compiled by MuJoCo to resolve defaults, includes, mesh transforms, and
orientation conventions. Static-body fusion and visual discarding are disabled
to retain the source parts. Default poses are extracted without stepping the
simulation. World geometry is kept separately in `robot.metadata['world_geometry']`.
Actuator, tendon, and equality-constraint counts are recorded; their behavior and
additional constraints are not imported. Heightfields, flexes, and plugin-defined
shapes are outside this initial rigid-geometry implementation.

USD reads default-time geometry and standard `UsdPhysics` schemas. Physics
definitions must be authored in the file; an arbitrary mesh-only USD file is not
a robot. Time samples, skeletal animation, and PhysX-specific extensions are
outside this version. Supported shapes are meshes, boxes, spheres, capsules,
cylinders, and cones. Unsupported robot geometry raises an error.

Xacro files must be expanded to URDF before loading.
The bundled FR3 URDF is already expanded from the manufacturer's included source.
To regenerate it without installing ROS:

```bash
.venv/bin/python -m pip install -e '.[assets]'
.venv/bin/python scripts/generate_fr3_urdf.py
```

The complete MJCF is included directly. The native MuJoCo gripper actuator is
`fr3_actuator8`: control 0 closes it, and 255 opens it fully.

The FR3 MJCF starts in its collision-free home pose, with the gripper open.
Its body reference transforms and joint `ref` values encode that pose while
preserving the original joint coordinates and limits. Finger joints explicitly
declare `type="slide"`, with positions measured in meters. For MJCF, motion
relative to the loaded geometry is `q - joint.metadata['reference_position'][0]`
for a scalar joint. The loader reads this default pose without stepping physics.

The URDF retains the manufacturer's zero-coordinate pose; it has no standard
initial joint-state field. Its all-zero pose is not the arm's home configuration.

## Code formatting

Use two spaces per indentation level, never tab characters. EditorConfig and
Ruff are configured to preserve this convention, including for Python.

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff format yarss scripts
```
