# yarss

Yet Another Robotics Simulation System. This directory is the installable
Python package and contains its source code, `pyproject.toml`, and installer.
Examples and robot assets live in the outer repository's `example/` directory.
Only MJCF loading is implemented. URDF and USD files raise `NotImplementedError`.

From this directory, install with:

```bash
./install.sh
```

The installer uses your current Python 3 installation. MuJoCo, NumPy, and
PyVista install automatically as dependencies. Python 3.10+ is required.

Load an MJCF robot and preview all its parts:

```python
from yarss import load_robot
from yarss.viewer import Viewer

robot = load_robot("/path/to/robot.xml")
viewer = Viewer()
viewer.add_robot(robot)
viewer.show()
```

The viewer displays every part's visual geometry at the loaded pose, including
the arm and gripper, with orange wireframes for collision shapes. Visual surfaces
are translucent so the collision geometry is visible inside them.
Toggle the overlay with the **Collision meshes** checkbox; hiding
it restores opaque visual surfaces. To start with collisions hidden, call
`viewer.show(show_collisions=False)`. PyVista supplies the default window size.
MJCF hinges have blue axis arrows and sliders in degrees; sliding joints have
green arrows and sliders in meters. Each slider shows the joint name and range.
Joint angles, initial angles, and angular limits are stored in degrees, so the
slider values can be passed directly to `Robot.set_joint_position()`.
Dragging edits the child's local transform and propagates world poses through
its descendants. Fixed joints have no slider. Ball and floating joints are not
controlled in this first version. Physics and coupled-joint constraints are
future work; the FR3 finger sliders operate independently.
