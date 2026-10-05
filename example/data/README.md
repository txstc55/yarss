# Example asset sources

The MJCF example describes the **Franka Research 3 (FR3)** seven-axis arm with
the **Franka Hand parallel gripper**. It includes visual meshes and collision
geometry for the base, seven arm links, hand, and two sliding fingers. Each
finger travels 0–0.04 m, giving an opening of up to 0.08 m.

The assets were downloaded on 2026-09-29. Original mesh and license files are
unchanged. The combined robot description is adapted as described below.
Asset licenses apply independently of YARSS code.

| Model | Upstream | License |
| --- | --- | --- |
| Franka Research 3 | [MuJoCo Menagerie FR3](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/franka_fr3) | Apache-2.0; [license](mjcf/franka_fr3/LICENSE) |
| Included Franka Hand | [MuJoCo Menagerie hand](https://github.com/google-deepmind/mujoco_menagerie/blob/4d038b3feae26ec82b46a4d586379114012a8ac7/franka_emika_panda/hand.xml) | Apache-2.0; [license](mjcf/franka_fr3/LICENSE.franka_hand) |

Source links pin the upstream commit. All referenced meshes are bundled;
loading requires no network access.

## Complete arm and gripper

Load **`mjcf/franka_fr3/fr3.xml`** for the robot, or
**`mjcf/franka_fr3/scene.xml`** for the robot with its floor and viewing setup.

Upstream commit: `4d038b3feae26ec82b46a4d586379114012a8ac7`.
`fr3.xml` contains the arm, wrist-mounted hand, both fingers, all joint and
collision definitions, the gripper tendon and equality constraint, and all
eight actuators. All meshes are in `assets/`. The entire `franka_fr3/` folder
can be copied and used independently.

The full model combines Menagerie's `franka_fr3/fr3.xml` and
`franka_emika_panda/hand.xml`. The hand is a child of `fr3_link7`, with a 0.107 m
translation along wrist Z and a -45-degree rotation about Z. Its names receive
the `fr3_` prefix to avoid collisions with arm definitions. Both source licenses
are included in the folder.

The gripper actuator is `fr3_actuator8`. Its upstream control range is 0–255:
0 closes the fingers and 255 opens them. The `home` keyframe retains the arm
pose and adds fully open fingers.

The model's reference body transforms are posed at home, and each moved joint
has a matching `ref` value. This makes the initial model collision-free while
preserving its original joint coordinates, motion, limits, and actuator targets.
Each finger joint explicitly declares `type="slide"`, `axis="0 1 0"`, and
`range="0 0.04"`. These limits are meters. The right finger's rotated body frame
makes its positive slide direction opposite the left finger.

YARSS imports geometry and joints and records actuator, tendon, and equality
constraint counts. It does not implement gripper coupling or actuator behavior.
The loader reads the default pose, which coincides with home. Loaded joint angles
and viewer sliders use degrees; sliding distances use meters.
