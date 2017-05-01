
def t_shape(tensor):
  """
  Returns:
    The shape of a tensorflow object in the format of numpy
  """
  return tuple([dimension.value for dimension in tensor.get_shape()])