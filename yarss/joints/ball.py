from .joint import Joint


class BallJoint(Joint):
  """Allow rotation in all directions about one point (3 DoF; 'ball')."""

  # Future: keep anchors coincident; apply swing/twist or cone limits if given.
  kind = "ball"
  dof = 3
