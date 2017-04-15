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


class DeepQNetworkState():

  """
    attributes:
      screens 
  """
  
  """s1 - s3 are OpenAIGym Boxes"""
  def __init__(self,s0,s1,s2,s3):
    pass 





