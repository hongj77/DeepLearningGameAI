from scipy import misc
import scipy
import numpy as np
import constants as C

def t_shape(tensor):
  """
  Returns:
    The shape of a tensorflow object in the format of numpy
  """
  return tuple([dimension.value for dimension in tensor.get_shape()])

def preprocess(state):
  """
  Params:
    state - takes in (h,w,3) 

  Returns:
    preprocessed state that resizes the image and grayscales it
    returns an image with shape (C.net_height, C.net_width, 1)
  """
  sg = convert_to_grayscale(state)
  #Google used bilinear 
  sg = scipy.misc.imresize(sg,(110,84),interp="bilinear")

  #crop to 84 x 84 which is most of game play screen 
  sg = sg[(110-84):,:]

  assert sg.shape == (C.net_height, C.net_width)
  return sg[:,:,np.newaxis]

def convert_to_grayscale(image):
  """
  Params:
    state - shape (h,w,3) the 3 represents r,g,b
  Returns:
    grayscaled image of (h,w,3)
  """
  grayed = np.dot(image[...,:3], [0.299, 0.587, 0.114])
  return grayed