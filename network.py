import numpy as np
import tensorflow as tf
import random

class DeepQNetwork:
  """
    attributes:
      replayMemory - [(s,a,r,s')] array of 4-tuple representing replay memory 
  """
  
  def __init__(self): 
  # initializes the layers of the CNN
    self.replay_memory = []

  def insert_tuple_into_replay_memory(self, mem_tuple):
    assert len(mem_tuple) == 4
    assert type(mem_tuple) == tuple
    self.replay_memory.append(mem_tuple)

  def sample_random_replay_memory(self, num_samples):
    assert num_samples <= len(self.replay_memory)
    assert num_samples >= 0 
    return random.sample(self.replay_memory, num_samples)

  def replay_memory_size(self):
    return len(self.replay_memory)
    
  def _createNetwork(): pass

  # trains the network using batches of replay memory
  def train(self, transition): pass

  def predict(self,state): pass


class DeepQNetworkState:

  """
    attributes:
      screens - [h,w,4] array
      s0 - state 1 converted to grayscale 
      s1 - state 2 converted to grayscale 
      s2 - state 3 converted to grayscale 
      s3 - state 4 converted to grayscale 
  """
  
  """s1 - s3 are OpenAIGym Boxes"""
  def __init__(self,s0,s1,s2,s3):
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.screens = np.concatenate((s0,s1,s2,s3), axis=2)

  @staticmethod
  def prepare(state):
    sg = DeepQNetworkState.convert_to_grayscale(state)
    return sg[:,:,np.newaxis]

  @staticmethod
  def convert_to_grayscale(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])





