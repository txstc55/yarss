# Example asset sources

The URDF and MJCF examples describe the real **Franka Research 3 (FR3)** seven-axis
robot arm with the **Franka Hand parallel gripper**. Both include visual meshes
and collision geometry for the base, seven arm links, hand, and two sliding
fingers. Each finger travels 0–40 mm, giving an opening of up to 80 mm.

The arm assets were downloaded on 2026-09-29. Meshes, URDF source configuration,
and license files are copied without modification. The complete URDF and MJCF
robot descriptions are derived from those sources as described below.
Asset licenses apply independently of YARSS code.

| Format | Model | Upstream | License |
| --- | --- | --- | --- |
| URDF | Franka Research 3 | [Franka Robotics description](https://github.com/frankarobotics/franka_description/tree/7aeeddc449edf8d62b594f9e36a81da53e7796f9) | Apache-2.0; [license](urdf/franka_description/LICENSE), [notice](urdf/franka_description/NOTICE) |
| MJCF | Franka Research 3 | [MuJoCo Menagerie FR3](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/franka_fr3) | Apache-2.0; [license](mjcf/franka_fr3/LICENSE) |
| MJCF | Included Franka Hand | [MuJoCo Menagerie hand](https://github.com/google-deepmind/mujoco_menagerie/blob/4d038b3feae26ec82b46a4d586379114012a8ac7/franka_emika_panda/hand.xml) | Apache-2.0; [license](mjcf/franka_fr3/LICENSE.franka_hand) |
| USD | Double cart-pole | [newton-physics/newton](https://github.com/newton-physics/newton/blob/3d5de3fa30c05be0db325c9de9f97bdeeca64551/newton/examples/assets/cartpole.usda) | Apache-2.0; [local license](usd/LICENSE.md) |

Source links pin the exact upstream commits. Downloaded files were verified
against their upstream Git blob hashes. All referenced meshes are bundled;
loading requires no network access.

## URDF

Load **`urdf/franka_description/urdf/fr3.urdf`**.

Upstream commit: `7aeeddc449edf8d62b594f9e36a81da53e7796f9`.
The subset under `urdf/franka_description/` preserves the ROS package layout:

- `robots/common/`: shared Xacro definitions.
- `robots/fr3/`: FR3 Xacro files and YAML kinematics, dynamics, inertials,
  accelerometer locations, and joint limits.
- `meshes/robots/fr3/visual/link0.dae` through `link7.dae`: visual meshes.
- `meshes/robots/fr3/collision/link0.stl` through `link7.stl`: collision meshes.
- `end_effectors/common/` and `end_effectors/franka_hand/`: gripper Xacro and inertials.
- `meshes/robot_ee/franka_hand_white/`: hand and finger meshes.
- `package.xml`, `README.rst`, `LICENSE`, and `NOTICE`: upstream package information.

The generated URDF uses `hand=true`, `ee_id=franka_hand`, and `with_sc=false`.
The last option omits auxiliary self-collision helper links; each physical arm
link still has its normal collision mesh. Manufacturer inertial data, joint
origins, axes, limits, and fixed sensor frames are retained.
The second prismatic finger joint mimics the first with multiplier 1 and offset 0;
their differently oriented frames make them slide in opposite physical directions.

From the YARSS repository, regenerate the included URDF with:

```bash
.venv/bin/python -m pip install -e '.[assets]'
.venv/bin/python scripts/generate_fr3_urdf.py
```

The script expands Xacro 2.1.1 using temporary copies with ROS package lookups
resolved locally. It preserves the upstream sources and the generated URDF's
`package://franka_description/...` mesh references. ROS is not required.
Normal loading uses the generated file and does not require Xacro.

## MJCF

Load **`mjcf/franka_fr3/fr3.xml`** for the robot, or
**`mjcf/franka_fr3/scene.xml`** for the robot with its floor and viewing setup.

Upstream commit: `4d038b3feae26ec82b46a4d586379114012a8ac7`.
`mjcf/franka_fr3/` is one complete robot package. `fr3.xml` directly contains
the arm, wrist-mounted hand, both fingers, all joint and collision definitions,
the gripper tendon and equality constraint, and all eight actuators. It does
not load an external hand model. All arm and gripper meshes are in `assets/`.
The entire folder can be copied and used independently.

The full model is adapted from Menagerie's `franka_fr3/fr3.xml` and
`franka_emika_panda/hand.xml` at the pinned commit. The hand is embedded as a
child of `fr3_link7`, with a 107 mm translation and -45-degree Z rotation,
matching the manufacturer URDF. Its names receive the `fr3_` prefix to avoid
collisions with arm definitions. The original mesh files are unchanged.
Both source licenses are included in the FR3 folder.

The gripper actuator is `fr3_actuator8`. Its upstream control range is 0–255:
0 closes the fingers and 255 opens them. The `home` keyframe retains the arm
pose and adds fully open fingers. `scene.xml` includes the complete robot,
floor, and viewing setup.

The MJCF's reference body transforms are posed at home, and each moved joint
has a matching `ref` value. This makes the initial model collision-free while
preserving its original joint coordinates, motion, limits, and actuator targets.
Each finger joint explicitly declares `type="slide"`, `axis="0 1 0"`, and
`range="0 0.04"`. These limits are meters, not radians. The right finger's
rotated body frame makes its positive slide direction opposite the left finger.

YARSS currently imports geometry and joints and records actuator, tendon, and
equality-constraint counts. It does not implement the gripper coupling or actuator
behavior. The loader reads the default pose, which now coincides with home
for the MJCF. The URDF still uses its standard zero-coordinate pose.

The URDF and MJCF have matching kinematics at the same joint coordinates but
different reference poses and upstream revisions. Their limits are not identical. The URDF
also has fixed accelerometer, flange, and tool-center frames absent from the MJCF. These
differences are preserved in the bundled examples.

## USD

The existing USD example was downloaded on 2026-09-28.
Upstream commit: `3d5de3fa30c05be0db325c9de9f97bdeeca64551`.
`newton/examples/assets/cartpole.usda` is stored as `usd/cartpole.usda`, along
with the repository's `LICENSE.md`. It is self-contained and includes four
rigid parts, a world-fixed base, one prismatic joint, and two continuous hinges.
Standard `UsdPhysics` definitions can be loaded without installing Isaac Sim.
