"""A general PyVista viewer with meshes and direct MJCF joint pose controls."""

from functools import partial

import numpy as np
import pyvista as pv

from ..joints import Joint
from ..parts import Geometry, Part
from ..robot import Robot


class Viewer:
  """Collect meshes, then render them with PyVista.

  Construct an empty viewer, call add_robot(), then show(). The mesh list can
  also contain other PyVista meshes. Robot loading happens outside the viewer.
  """

  def __init__(self, off_screen: bool = False):
    self.meshes: list[pv.DataSet] = []
    self.collision_meshes: list[pv.DataSet] = []
    self.plotter = pv.Plotter(off_screen=off_screen)
    self.robots: list[Robot] = []
    # Robot meshes stay in part coordinates; actors carry their world transforms.
    self._mesh_parts: dict[int, Part] = {}
    self._part_actors: dict[Part, list[pv.Actor]] = {}
    self._joint_axes: list[tuple[Robot, Joint, pv.Actor, float]] = []
    self._joint_label_points: pv.PolyData | None = None

  def add_robot(self, robot: Robot) -> None:
    """Append static visual and collision meshes in their owning part's coordinates."""
    self.robots.append(robot)
    for part in robot.parts.values():
      for geometries, meshes in (
        (part.visuals, self.meshes),
        (part.collisions, self.collision_meshes),
      ):
        for geometry in geometries:
          mesh = _geometry_mesh(geometry)
          mesh.transform(geometry.transform, inplace=True)
          meshes.append(mesh)
          self._mesh_parts[id(mesh)] = part

  def show(self, show_collisions: bool = True) -> None:
    """Draw meshes, collision visibility, and degree/meter sliders for MJCF joints."""
    visual_actors = []
    collision_actors = []
    self._part_actors.clear()
    for index, mesh in enumerate(self.meshes):
      part = self._mesh_parts.get(id(mesh))
      actor = self.plotter.add_mesh(
        mesh,
        name=f"mesh_{index}",
        color="#aebdcb",
        smooth_shading=True,
        user_matrix=part.transform if part is not None else None,
      )
      visual_actors.append(actor)
      if part is not None:
        # Keep PyVista's shaded output as static data, detached from its normal filter.
        actor.mapper.dataset = actor.mapper.dataset
        self._part_actors.setdefault(part, []).append(actor)
    for index, mesh in enumerate(self.collision_meshes):
      part = self._mesh_parts.get(id(mesh))
      actor = self.plotter.add_mesh(
        mesh,
        name=f"collision_{index}",
        color="#d65c00",
        style="wireframe",
        line_width=1.5,
        user_matrix=part.transform if part is not None else None,
      )
      collision_actors.append(actor)
      if part is not None:
        self._part_actors.setdefault(part, []).append(actor)

    if collision_actors:

      def set_collision_visibility(visible: bool) -> None:
        for actor in collision_actors:
          actor.visibility = visible
        for actor in visual_actors:
          # Restore solid surfaces when the collision overlay is hidden.
          actor.prop.opacity = 0.3 if visible else 1.0
        self.plotter.render()

      set_collision_visibility(show_collisions)
      self.plotter.add_checkbox_button_widget(
        set_collision_visibility,
        value=show_collisions,
        size=25,
        border_size=3,
        color_on="#d65c00",
        color_off="#f3f5f7",
        background_color="#f3f5f7",
      )
      self.plotter.add_text("Collision meshes", position=(45, 14), font_size=10, color="#333333")
    self._add_joint_controls()
    self.plotter.set_background("#f3f5f7")
    self.plotter.show()

  def _add_joint_controls(self) -> None:
    """A hinge has one angular slider; a slide has one linear slider; fixed has none."""
    # first we get all the controllable joints from all robots, then we create a slider for each joint
    controls = [
      (robot, joint)
      for robot in self.robots
      if robot.metadata.get("format") == "mjcf"
      for joint in robot.joints.values()
      if joint.kind in {"rotating", "unlimited_rotating", "sliding"}
    ]
    if not controls:
      return

    for robot, joint in controls:
      positions = np.array([part.transform[:3, 3] for part in robot.parts.values()])
      length = max(float(np.linalg.norm(np.ptp(positions, axis=0))), 0.1) * 0.25
      axis_mesh = pv.Arrow(direction=joint.axis, scale=length, shaft_radius=0.02, tip_radius=0.07)
      frame = robot.joint_world_frame(joint.name)
      color = "#16803c" if joint.kind == "sliding" else "#2563eb"
      actor = self.plotter.add_mesh(
        axis_mesh,
        name=f"joint_axis_{len(self._joint_axes)}",
        color=color,
        lighting=False,
        user_matrix=frame,
      )
      self._joint_axes.append((robot, joint, actor, length))

    self._joint_label_points = pv.PolyData(np.zeros((len(controls), 3)))
    self._joint_label_points.point_data["joint_name"] = [joint.name for _, joint in controls]
    self._update_joint_axes()
    self.plotter.add_point_labels(
      self._joint_label_points,
      "joint_name",
      font_size=12,
      text_color="#243443",
      shape_color="white",
      shape_opacity=0.8,
      show_points=False,
      always_visible=True,
      reset_camera=False,
      name="joint_names",
    )

    spacing = min(0.1, 0.8 / max(len(controls) - 1, 1))
    for index, (robot, joint) in enumerate(controls):
      linear = joint.kind == "sliding"
      unit = "m" if linear else "deg"
      # An unbounded joint still needs a finite slider; this is only a preview range.
      span = 0.1 if linear else 180.0
      limit = joint.limits.get("value")
      lower = limit.lower if limit and limit.lower is not None else joint.value - span
      upper = limit.upper if limit and limit.upper is not None else joint.value + span
      decimals = 3 if linear else 1
      title = f"{joint.name} ({unit})"
      if limit is None:
        title += " (preview)"
      y = 0.9 - index * spacing
      widget = self.plotter.add_slider_widget(
        partial(
          self._set_joint_value, robot, joint
        ),  # fix the first two arguments as this robot, this joint
        (lower, upper),
        value=joint.value,
        title="",
        pointa=(0.79, y),
        pointb=(0.92, y),
        color="#16803c" if linear else "#2563eb",
        slider_width=0.012,
        tube_width=0.003,
        fmt=f"%.{decimals}f {unit}",
        interaction_event="always",
      )
      representation = widget.GetRepresentation()
      representation.SetLabelHeight(0.016)
      representation.SetSliderLength(0.02)
      representation.SetEndCapWidth(0.018)
      representation.SetEndCapLength(0.002)
      # The native title sits below the track; place our labels explicitly instead.
      for label_index, (text, position, alignment) in enumerate(
        (
          (title, (0.855, y + 0.052), "center"),
          (f"{lower:.{decimals}f}", (0.78, y), "right"),
          (f"{upper:.{decimals}f}", (0.93, y), "left"),
        )
      ):
        label = self.plotter.add_text(
          text,
          position=position,
          viewport=True,
          font_size=7,
          color="#243443",
          name=f"joint_control_{index}_{label_index}",
          render=False,
        )
        label.prop.justification_horizontal = alignment
        label.prop.justification_vertical = "center"

  def _set_joint_value(self, robot: Robot, joint: Joint, value: float) -> None:
    """Slider -> child local pose -> descendant world poses -> displayed meshes."""
    # PyVista also calls the callback when it first creates each slider.
    if np.isclose(value, joint.value, atol=1e-12, rtol=0):
      return
    updated = set(robot.set_joint_value(joint, value))
    for part in updated:
      for actor in self._part_actors.get(part, ()):
        # VTK transforms the static mesh and its lighting normals while rendering.
        actor.user_matrix = part.transform
    self._update_joint_axes(updated)
    self.plotter.reset_camera_clipping_range()
    self.plotter.render()

  def _update_joint_axes(self, updated: set[Part] | None = None) -> None:
    if self._joint_label_points is None:
      return
    points = self._joint_label_points.points.copy()
    for index, (robot, joint, actor, length) in enumerate(self._joint_axes):
      if updated is not None and joint.child not in updated:
        continue
      frame = robot.joint_world_frame(joint.name)
      actor.user_matrix = frame
      direction = frame[:3, :3] @ joint.axis
      points[index] = frame[:3, 3] + length * direction
    self._joint_label_points.points = points


def _geometry_mesh(geometry: Geometry) -> pv.PolyData:
  """Make a mesh in geometry coordinates; source arrays remain unchanged."""
  if geometry.kind == "mesh":
    faces = pv.CellArray.from_irregular_cells(geometry.faces)
    return pv.PolyData(geometry.vertices, faces=faces, deep=True)

  parameters = geometry.parameters
  if geometry.kind == "box":
    x, y, z = parameters["size"]
    return pv.Cube(x_length=x, y_length=y, z_length=z)
  if geometry.kind == "sphere":
    return pv.Sphere(radius=parameters["radius"])
  if geometry.kind == "ellipsoid":
    return pv.ParametricEllipsoid(*parameters["radii"])

  # MJCF cylinders and capsules point along local Z.
  if geometry.kind == "cylinder":
    return pv.Cylinder(
      direction=(0, 0, 1), radius=parameters["radius"], height=parameters["length"]
    )
  if geometry.kind == "capsule":
    return pv.Capsule(
      direction=(0, 0, 1), radius=parameters["radius"], cylinder_length=parameters["length"]
    )
  raise ValueError(f"Unsupported MJCF geometry type {geometry.kind!r}")
