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

class Replay():
  def __init__(self):
    self.replay_memory = []
    self.batch_size = C.ai_batch_size

  def insert_tuple_into_replay_memory(self, mem_tuple):
    """
    Params:
      mem_tuple - (network_state, action, reward, new_network_state, done)
    """
    assert len(mem_tuple) == 5
    assert type(mem_tuple) == tuple

    self.replay_memory.append(mem_tuple)
    
    # if memory size is full, dump the first 1000
    if self.replay_memory_size() > C.ai_replay_mem_total_size:
      self.replay_memory = self.replay_memory[1000:]

  def sample_random_replay_memory(self, num_samples):
    assert num_samples <= len(self.replay_memory)
    assert num_samples >= 0
    return random.sample(self.replay_memory, num_samples)


  def get_minibatch(self):
    """
    Returns:
      prestates, actions, rewards, poststates, terminals
      prestates and poststates are (32,4,84,84)
    """
    transitions = random.sample(self.replay_memory, self.batch_size)

    states = np.ndarray((self.batch_size,84,84,4))
    new_states = np.ndarray((self.batch_size,84,84,4))
    rewards = np.ndarray(self.batch_size) # shape (32,)
    actions = np.ndarray(self.batch_size) # shape (32,)
    terminals = np.ndarray(self.batch_size)

    for i in range(self.batch_size):
      s, a, r, ns, d = transitions[i]
      states[i,:,:,:] = s.screens()[:,:,:]
      new_states[i,:,:,:] = ns.screens()[:,:,:]
      rewards[i] = r
      actions[i] = a
      terminals[i] = d

    states = np.moveaxis(states,3,1)
    new_states = np.moveaxis(new_states,3,1)

    assert states.shape == (self.batch_size, 4, 84, 84)
    assert new_states.shape == (self.batch_size, 4, 84, 84)

    return states, actions, rewards, new_states, terminals

  def replay_memory_size(self):
    return len(self.replay_memory)