from .joint import Joint


class CylindricalJoint(Joint):
  """Allow independent sliding and rotation along one shared axis (2 DoF)."""

  # Future: constrain sideways movement and tilt; limit slide and spin.
  kind = "cylindrical"
  dof = 2
