import numpy as np
import os
import numpy as np
import tensorflow as tf
from tensorflow import contrib
import random
import scipy
from scipy import misc
import constants as C
import matplotlib.pyplot as plt
import pdb
from utils import *
from state import *

class DeepQNetworkState:
  """
  Representation of a single state in our DeepQNetwork. Each state consists of four game screens in order to account for velocity. 

    attributes:
      screens - [h,w,4] array representing the last 4 game screens
      s0 - state 1 converted to grayscale 
      s1 - state 2 converted to grayscale 
      s2 - state 3 converted to grayscale 
      s3 - state 4 converted to grayscale 

      s0 - s3 are numpy.ndarray of shape (C.net_height, C.net_width, 1) 
  """
  def __init__(self,s0,s1,s2,s3):
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3

  def screens(self):
    return np.concatenate((self.s0,self.s1,self.s2,self.s3), axis=2)