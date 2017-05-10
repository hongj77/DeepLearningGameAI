import numpy as np
import random
import constants as C
import pdb

class ReplayMemory:
  def __init__(self):
    # 1000000
    self.size = C.ai_replay_mem_total_size

    # preallocate memory
    self.actions = np.empty(self.size, dtype = np.uint8)
    self.rewards = np.empty(self.size, dtype = np.integer)

    # (1000000, 84, 84)
    self.screens = np.empty((self.size, C.net_height, C.net_width), dtype = np.uint8)

    self.terminals = np.empty(self.size, dtype = np.bool)
    self.history_length = C.net_num_screens

    # (84,84)
    self.dims = (C.net_height, C.net_width)

    # 32
    self.batch_size = C.ai_batch_size

    self.count = 0 # how many elements
    self.current = 0 # current index

    # (32,4,84,84)
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)

    print("Replay memory size: %d" % self.size)

  def add(self, action, reward, screen, terminal):
    assert action >= 0 and action < C.net_n_actions
    assert screen.shape == self.dims

    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1) # either full or not full yet
    self.current = (self.current + 1) % self.size # always advance and then mod

    # print("Memory count %d" % self.count)
    # print("Memory current %d" % self.current)

  
  def getState(self, index):
    # returns (4, 84, 84)
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def getLast(self):
    """
    Gets the last four screens in the screens
    """
    assert self.current > self.history_length
    # Note: current is already 1 ahead of the index you want to get
    last = self.screens[self.current-self.history_length:self.current]
    assert last.shape == (C.net_num_screens, C.net_height, C.net_width)
    return last


  def getMinibatch(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    assert actions.shape == (self.batch_size,)
    assert rewards.shape == (self.batch_size,)
    assert terminals.shape == (self.batch_size,)
    assert self.prestates.shape == (self.batch_size, self.history_length, C.net_height, C.net_width)
    assert self.poststates.shape == (self.batch_size, self.history_length, C.net_height, C.net_width)

    return self.prestates, actions, rewards, self.poststates, terminals
