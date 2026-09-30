from .joint import Joint


class DistanceJoint(Joint):
  """Keep anchor separation fixed or within a distance interval."""

  # Future: constrain distance only; rotation remains free. Active limits
  # determine the DoF, so there is no single fixed count for every instance.
  kind = "distance"
  dof = None
