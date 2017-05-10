import numpy as np
import tensorflow as tf
from tensorflow import contrib
import random
import scipy
from scipy import misc
import constants as C
import pdb

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
    returns an image with shape (C.net_height, C.net_width)
  """
  sg = convert_to_grayscale(state)
  #Google used bilinear 
  sg = scipy.misc.imresize(sg,(110,84),interp="bilinear")

  #crop to 84 x 84 which is most of game play screen 
  sg = sg[(110-84):,:]

  assert sg.shape == (C.net_height, C.net_width)
  return sg

def convert_to_grayscale(image):
  """
  Params:
    state - shape (h,w,3) the 3 represents r,g,b
  Returns:
    grayscaled image of (h,w,3)
  """
  grayed = np.dot(image[...,:3], [0.299, 0.587, 0.114])
  return grayed

def initialize_state(state):
  """
  For the beginning of the game, given a game state (84,84)
  Returns:
    A zero filled image including the state of size (4,84,84)
  """
  assert state.shape == (C.net_height, C.net_width)
  result = np.zeros(C.net_num_screens, C.net_height, C.net_width)
  result[0, :] = state[:,:]
  assert result.shape == (C.net_num_screens, C.net_height, C.net_width)
  return result