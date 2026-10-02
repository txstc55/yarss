# YARSS — Yet Another Robotics Simulation System

This first version loads robot descriptions into a shared `Robot` containing
`Part` objects and typed `Joint` connections. It extracts geometry, default
poses, joint frames, axes, and limits. Joint motion and physics are placeholders.

## Setup and MJCF preview

From the outer repository folder:

```bash
./yarss/install.sh
python3 example/viewer/view_robot.py
```

The installer installs the single package `yarss` into your current Python 3
installation. It does not create a virtual environment. MuJoCo and PyVista are
standard dependencies and install automatically. Python 3.10+ is required.

The inner `yarss/` folder is the installable package: it contains the Python code,
`pyproject.toml`, and `install.sh`. The installer runs `python3 -m pip install -e .`
from that folder. Editable mode makes code edits take effect without reinstalling.
You import the installed package with `import yarss`. The second command opens
the complete FR3 arm and gripper.

URDF uses Python's XML reader, trimesh for mesh files,
and pycollada for COLLADA geometry. NumPy handles matrices and mesh vertices.
The optional `usd` dependency group adds OpenUSD for USD loading.
The example assets are included and load offline after installation.

| Bundled example | Parts | Joints |
| --- | ---: | ---: |
| Franka Research 3 with parallel gripper, manufacturer URDF | 26 | 25 |
| Franka Research 3 with parallel gripper, Menagerie-based MJCF | 12 | 12 |
| Newton cart-pole, USD | 4 | 4 |

Both FR3 examples contain seven rotating arm joints and the Franka Hand's two
sliding finger joints. Each finger slides from 0 to 40 mm; the fingers move
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
yarss/                     # Repository
├── yarss/                 # Installable Python package
│   ├── install.sh
│   ├── pyproject.toml
│   ├── README.md
│   ├── __init__.py
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
│   ├── viewer/viewer.py   # General PyVista viewer; add_robot adds every part's visuals
│   └── transforms.py      # Static pose conversion helpers
└── example/
    ├── data/              # Freely licensed robot and gripper assets
    └── viewer/
        └── view_robot.py  # Load an MJCF robot and preview all its parts
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

## Viewing an MJCF robot

After the setup above, open the bundled FR3 arm and gripper in PyVista:

```bash
python3 example/viewer/view_robot.py
```

Drag to orbit and scroll to zoom. Every part is shown at its loaded world pose:
visual geometry uses translucent gray surfaces, and collision shapes use orange
wireframes. Use the **Collision meshes** checkbox in the bottom-left corner to
toggle the collision overlay. Hiding it restores opaque visual surfaces.
Joints are not drawn. To load another MJCF file:

```bash
python3 example/viewer/view_robot.py /path/to/robot.xml
```

`Viewer()` creates a PyVista plotter with its default window size and empty mesh
lists. Load the robot separately, add it to the viewer, then render:

```python
from yarss import load_robot
from yarss.viewer import Viewer

robot = load_robot("example/data/mjcf/franka_fr3/fr3.xml")
viewer = Viewer()
viewer.add_robot(robot)
viewer.show()
```

`add_robot()` appends the visual geometry from every part to `viewer.meshes`.
It appends collision geometry to `viewer.collision_meshes`. It preserves meshes
already in either list. The viewer stores no single robot; other PyVista meshes
can also be appended in world coordinates. `show()` renders both lists, making
the visual surfaces translucent while collision shapes are visible. Use
`viewer.show(show_collisions=False)` to start with the collision overlay hidden;
the checkbox can still turn it on.

Parts without geometry, such as the FR3's empty root frame, add no meshes.
Joint definitions are not used to draw anything; each part already has its world
transform from the loader. Joint visualization and motion are separate work.

For each geometry, the viewer computes:

```python
world = part.transform @ geometry.transform
world_vertices = geometry.vertices @ world[:3, :3].T + world[:3, 3]
```

The second line applies the matrix to an entire `N x 3` array of vertices.
`Part.transform` already includes all ancestors, so their matrices must not be
applied again. The viewer leaves the source geometry and transforms unchanged.
It draws `Part.visuals` in a neutral color; source materials are not imported.
For MJCF, that list follows MuJoCo's [default visible groups 0, 1, and 2](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-group).
The FR3's separate collision shapes use group 3 and are stored in `Part.collisions`;
the viewer draws these as orange wireframes, including the finger pad boxes.

## Joint classes

Every type directly subclasses `Joint`; each file describes its intended
constraints in comments. There are no constraint solvers or motion methods yet.

| Class | Intended motion | Imported from |
| --- | --- | --- |
| `FixedJoint` | No relative motion | URDF, implicit MJCF, USD |
| `RotatingJoint` | Limited rotation about one axis | URDF revolute, MJCF hinge, USD |
| `UnlimitedRotatingJoint` | Unlimited rotation about one axis | URDF continuous, unlimited MJCF/USD hinge |
| `SlidingJoint` | Translation along one axis | URDF prismatic, MJCF slide, USD |
| `PlanarJoint` | Two translations and one rotation in a plane | URDF |
| `FloatingJoint` | Three translations and three rotations | URDF, MJCF free |
| `BallJoint` | Rotation about a common point | MJCF ball, USD spherical |
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
- Mesh vertices are NumPy arrays of shape `(N, 3)`. Faces are tuples of vertex
  indices, allowing variable-length polygons. URDF meshes also retain `mesh_path`.
  USD polygons remain polygons; primitives remain shape/dimension records.
- `Part.transform` maps part coordinates to world coordinates at the file's
  default pose. `Geometry.transform` maps geometry coordinates to part coordinates.
  Joint anchor matrices map the joint frame into each endpoint's part frame.
- Transforms are NumPy arrays of shape `(4, 4)` and use column vectors.
  `part.transform @ geometry.transform @ [x, y, z, 1]` gives a homogeneous
  world vertex. Joint axes are NumPy arrays of shape `(3,)`. Mesh scale is already
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
python3 -m pip install -e './yarss[dev]'
python3 -m ruff format --config yarss/pyproject.toml yarss example/viewer
```
