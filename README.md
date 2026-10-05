# YARSS — Yet Another Robotics Simulation System

This first version loads MJCF robot descriptions into a `Robot` containing
`Part` objects and typed `Joint` connections. It extracts geometry, default
poses, joint frames, axes, and limits. The viewer can edit MJCF hinge and slide
poses directly; physics and constraint solving are future work.

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

NumPy handles matrices and mesh vertices. MuJoCo resolves model files and their
assets. The bundled example loads offline after installation.

The example is a complete **Franka Research 3 (FR3)** arm with the **Franka Hand
parallel gripper**: 12 parts and 12 joints, including seven rotating arm joints,
two sliding finger joints, and three fixed connections. Each finger slides from
0 to 0.04 m, giving a maximum opening of 0.08 m. Visual and collision meshes are
included for the arm, hand, and fingers.

`fr3.xml` contains the full robot, equality constraint, tendon, and gripper
actuator. All meshes live together in `franka_fr3/assets/`. Coupling definitions
remain in the file; YARSS does not simulate them yet.
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
│   │   └── mjcf.py         # MuJoCo compiler and geometry extraction
│   ├── joints/
│   │   ├── joint.py        # Base Joint, limits, and mimic definitions
│   │   └── ...            # One file per joint subclass
│   ├── parts/part.py      # Part and Geometry
│   ├── robot/robot.py     # Parts and their joint graph
│   ├── viewer/viewer.py   # General PyVista viewer; add_robot adds every part's visuals
│   └── transforms.py      # Static pose conversion helpers
└── example/
    ├── check_units.py     # MJCF units and loader checks
    ├── data/              # Freely licensed robot and gripper assets
    └── viewer/
        ├── view_robot.py  # Load an MJCF robot and preview all its parts
        └── check_joint_controls.py
```

## Loading a robot

```python
from yarss import load_robot

robot = load_robot("example/data/mjcf/franka_fr3/fr3.xml")

print(robot.name)
print([part.name for part in robot.roots])

for joint in robot.joints.values():
  parent_name = joint.parent.name if joint.parent is not None else "world"
  print(joint.name, joint.kind, parent_name, "->", joint.child.name)

wrist = robot.parts["fr3_link7"]
mesh = wrist.visuals[0]
print(mesh.source, len(mesh.vertices), len(mesh.faces))
print([joint.name for joint in robot.connections("fr3_link7")])
```

The accompanying `example/data/mjcf/franka_fr3/scene.xml` includes the robot,
floor, lighting, and viewing settings. Finger joints are named
`fr3_finger_joint1` and `fr3_finger_joint2`.

`Loader().load(path)` is equivalent to `load_robot(path)`. Both accept strings
or `pathlib.Path` objects and support `.xml` and `.mjcf` files with a `<mujoco>`
root element. MuJoCo resolves includes and mesh paths relative to the model.
Each joint's `parent` and `child` refer directly to the `Part` objects in
`robot.parts`. World attachments use `None` as the joint's parent.

URDF and USD loading raise `NotImplementedError` for now. This includes URDF
content with a `<robot>` root in an XML file. The loader has no format-specific
options; use `load_robot(path)` or `Loader().load(path)`.

## Viewing an MJCF robot

After the setup above, open the bundled FR3 arm and gripper in PyVista:

```bash
python3 example/viewer/view_robot.py
```

Drag to orbit and scroll to zoom. Every part is shown at its loaded world pose:
visual geometry uses translucent gray surfaces, and collision shapes use orange
wireframes. Use the **Collision meshes** checkbox in the bottom-left corner to
toggle the collision overlay. Hiding it restores opaque visual surfaces.
Blue arrows mark hinge axes and green arrows mark sliding axes. Each movable
joint has a labeled slider with its range: arm angles are displayed in degrees,
and finger travel in meters. Dragging updates the child part and all descendants,
including their visual meshes, collision shapes, axes, and labels. Fixed joints
have no degree of freedom and no slider. To load another MJCF file:

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
For MJCF, `Part.local_transform` maps from the part to its parent; `Part.transform`
is the current world pose. A slider calls `Robot.set_joint_value()` to rebuild
the child's local pose from its joint anchors and coordinate, then propagate
world poses down the tree. Motion is measured from each joint's `initial_value`,
so the FR3 opens at its loaded home pose without a jump.
Joint angles and angular limits are stored in degrees, matching the sliders.
Several hinge/slide joints on one body compose in source order.
Parts store their connected parents, children, and incoming joints when a joint
is created. Pose updates follow those connections and return the affected parts;
they do not rebuild the whole robot's graph. The viewer updates only those parts'
meshes and joint axes, rotating cached normals and arrow vertices instead of
rebuilding them on every slider movement.

These controls edit poses directly. They do not apply forces, prevent collisions,
or enforce tendons, mimic relations, or equality constraints; the two finger
sliders operate independently. This first version controls hinges and slides;
ball and floating joints remain static. Unbounded joints get a finite preview
slider range, labeled as such, without adding physical limits.

Check the sliders, rendered meshes, and joint frames against native MuJoCo
kinematics without opening an interactive window:

```bash
python3 example/viewer/check_joint_controls.py
```

For each geometry, the viewer computes:

```python
world = part.transform @ geometry.transform
world_vertices = geometry.vertices @ world[:3, :3].T + world[:3, 3]
```

The second line applies the matrix to an entire `N x 3` array of vertices.
`Part.transform` already includes all ancestors, so their matrices must not be
applied again. The viewer preserves the source geometry and joint anchor frames;
sliders update joint values and part transforms.
It draws `Part.visuals` in a neutral color; source materials are not imported.
For MJCF, that list follows MuJoCo's [default visible groups 0, 1, and 2](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-group).
The FR3's separate collision shapes use group 3 and are stored in `Part.collisions`;
the viewer draws these as orange wireframes, including the finger pad boxes.

## Joint classes

Every type directly subclasses `Joint`; each file describes its intended
constraints in comments. Fixed, sliding, and rotating joints each implement
`motion_transform()` to return their own motion matrix. Rotations are calculated
directly from the axis and the angle in degrees using Rodrigues' formula.
Only the trigonometric calculation inside `rotation_transform()` converts that
angle to radians for NumPy's sine and cosine functions.
Other types remain placeholders and raise `NotImplementedError` for pose editing.
There are no constraint solvers yet.

Every joint uses the same state field names: `value` for its current state and
`initial_value` for the state loaded from the file. Their representation depends
on the joint type:

- `RotatingJoint` and `UnlimitedRotatingJoint`: a scalar in degrees.
- `SlidingJoint`: a scalar in meters.
- `BallJoint`: a 3x3 rotation matrix in joint coordinates.
- `FloatingJoint`: a 4x4 world transform with translations in meters.
- `FixedJoint`: zero; its motion transform is always identity.

The MJCF loader initializes both fields. `initial_value` stays unchanged during
pose editing; matrix values are separate copies. Scalar motion uses
`joint.value - joint.initial_value`. Ball and floating pose editing remains
unimplemented.

```python
joint = robot.joints["fr3_joint4"]
print(joint.initial_value, joint.value)
robot.set_joint_value(joint.name, -120.0)  # Degrees for this rotating joint.
```

| Class | Intended motion | MJCF source |
| --- | --- | --- |
| `FixedJoint` | No relative motion | Implicit body attachment |
| `RotatingJoint` | Limited rotation about one axis | Limited hinge |
| `UnlimitedRotatingJoint` | Unlimited rotation about one axis | Unlimited hinge |
| `SlidingJoint` | Translation along one axis | Slide |
| `FloatingJoint` | Three translations and three rotations | Free |
| `BallJoint` | Rotation about a common point | Ball |
| `PlanarJoint` | Two translations and one rotation in a plane | Placeholder |
| `DistanceJoint` | Fixed or bounded anchor separation | Placeholder |
| `D6Joint` | Individually free, limited, or locked axes | Placeholder |
| `CylindricalJoint` | Independent slide and spin on one axis | Placeholder |
| `UniversalJoint` | Rotation about two intersecting axes | Placeholder |
| `ScrewJoint` | Rotation coupled to translation by pitch | Placeholder |

MJCF bodies form a tree. Multiple joints on one body share endpoints and retain
their source order when computing that body's motion.
`joint.child_frame_inverse` is computed once during construction and reused
for pose updates and joint-axis drawing. The child frame and its cached inverse
are read-only NumPy arrays; slider movements change part poses, not these frames.

## Geometry and coordinate conventions

- `Part.visuals` and `Part.collisions` contain `Geometry` objects. Bodies without
  geometry are retained because joints can connect through them.
- Mesh vertices are NumPy arrays of shape `(N, 3)`. Faces contain triangle vertex
  indices. Primitives retain their shape and dimensions.
- `Part.transform` maps part coordinates to world coordinates. It starts at the
  file's default pose and changes when a joint moves. `Part.local_transform`
  stores the pose relative to the parent. `Geometry.transform` maps geometry
  coordinates to part coordinates. Joint anchor matrices map the joint frame
  into each endpoint's part frame.
- Transforms are NumPy arrays of shape `(4, 4)` and use column vectors.
  `part.transform @ geometry.transform @ [x, y, z, 1]` gives a homogeneous
  world vertex. Joint axes are NumPy arrays of shape `(3,)`. MuJoCo's compiled
  vertices and geometry poses already account for mesh scale and recentering.
- Lengths are in meters; joint angles and angular limits are in degrees.
  The loader converts compiled angular coordinates when loading.
  `rotation_transform()` and `rpy_quaternion()` accept degrees.
  Imported angular stiffness uses N*m/degree and linear stiffness uses N/m,
  preserving the physical torque or force for the same displacement.
- Scalar joint limits use `joint.limits['value']`. Ball limits use `'angle'`.
  Rotation matrices, quaternions, axes, and scale factors are dimensionless.

Run the MJCF unit and loader checks with:

```bash
python3 example/check_units.py
```

The separate viewer check above exercises every FR3 slider against native
MuJoCo kinematics. Source model files retain their authored angle convention;
YARSS joint coordinates use degrees after loading.

## Current scope

The loader extracts robot descriptions, and the viewer edits scalar joint poses
using forward kinematics. Forces, motors, contacts, and mimic relationships are
not simulated. Material rendering and texture data are not imported into `Geometry`.

MJCF is compiled by MuJoCo to resolve defaults, includes, mesh transforms, and
orientation conventions. Static-body fusion and visual discarding are disabled
to retain the source parts. Default poses are extracted without stepping the
simulation. World geometry is kept separately in `robot.metadata['world_geometry']`.
Actuator, tendon, and equality-constraint counts are recorded; their behavior and
additional constraints are not imported. Heightfields, flexes, and plugin-defined
shapes are outside this initial rigid-geometry implementation.

The complete MJCF is included directly. The native MuJoCo gripper actuator is
`fr3_actuator8`: control 0 closes it, and 255 opens it fully.

The FR3 MJCF starts in its collision-free home pose, with the gripper open.
Its body transforms and joint `ref` values encode that pose while
preserving the original joint coordinates and limits. Finger joints explicitly
declare `type="slide"`, with positions measured in meters. For MJCF, motion
relative to the loaded geometry is `joint.value - joint.initial_value` for
both rotating and sliding joints.
The loader reads this default pose without stepping physics.

## Code formatting

Use two spaces per indentation level, never tab characters. EditorConfig and
Ruff are configured to preserve this convention, including for Python.

```bash
python3 -m pip install -e './yarss[dev]'
python3 -m ruff format --config yarss/pyproject.toml yarss example
```
