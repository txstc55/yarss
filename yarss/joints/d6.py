from .joint import Joint


class D6Joint(Joint):
  """Individually free, limit, or lock each of six motion axes."""

  # Future: interpret each axis limit; absent = free, lower > upper = locked.
  # The configured axes, rather than the class, determine the available DoF.
  kind = "d6"
  dof = None
