"""A general PyVista viewer with visual and collision meshes in world coordinates."""

import pyvista as pv

from ..parts import Geometry
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

  def add_robot(self, robot: Robot) -> None:
    """Append every part's visual and collision geometry in world coordinates."""
    for part in robot.parts.values():
      for geometries, meshes in (
        (part.visuals, self.meshes),
        (part.collisions, self.collision_meshes),
      ):
        for geometry in geometries:
          mesh = _geometry_mesh(geometry)

          # Geometry -> part -> world. Part.transform already includes ancestors.
          world = part.transform @ geometry.transform
          # Vertices are rows, so transpose the matrix's rotation/scale block.
          mesh.points = mesh.points @ world[:3, :3].T + world[:3, 3]
          meshes.append(mesh)

  def show(self, show_collisions: bool = True) -> None:
    """Draw the meshes with a checkbox to toggle collision wireframes."""
    visual_actors = []
    collision_actors = []
    for index, mesh in enumerate(self.meshes):
      actor = self.plotter.add_mesh(
        mesh,
        name=f"mesh_{index}",
        color="#aebdcb",
        smooth_shading=True,
      )
      visual_actors.append(actor)
    for index, mesh in enumerate(self.collision_meshes):
      actor = self.plotter.add_mesh(
        mesh,
        name=f"collision_{index}",
        color="#d65c00",
        style="wireframe",
        line_width=1.5,
      )
      collision_actors.append(actor)

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
    self.plotter.set_background("#f3f5f7")
    self.plotter.show()


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
