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

class DeepQNetwork:
  def __init__(self): 
    self.replay_memory = [] # [(s,a,r,s',d)] array of 4-tuple representing replay memory 
    self.batch_size = C.ai_batch_size
    self.height = C.net_height
    self.width = C.net_width 
    self.num_screens = C.net_num_screens
    self.n_actions = C.net_n_actions
    self.discount_factor = C.net_discount_factor

    # a single sample of a game state that we hold on to for mean q-values over time
    self.validation_set = None
    self.validation = False # set this to true when validation set is initialized

    # PLACEHOLDERS 
    self.x = tf.placeholder(tf.float32, [None, self.height,self.width,self.num_screens])
    self.y = tf.placeholder(tf.float32, [None, 1]) # one max q value per batch sample
    self.actions_taken = tf.placeholder(tf.int32, [self.batch_size]) # values in range [0,5]
    assert t_shape(self.actions_taken) == (self.batch_size, )

    self.weights = {
      # 8x8 conv, 4 inputs (game screens) 32 outputs
      'wc1': tf.Variable(tf.truncated_normal([8,8,4,32], stddev = 0.01)),
      # 4x4 conv, 32 inputs 64 outputs
      'wc2': tf.Variable(tf.truncated_normal([4,4,32,64], stddev = 0.01)),
      # 3x3 conv, 64 inputs 64 outputs
      'wc3': tf.Variable(tf.truncated_normal([3,3,64,64], stddev = 0.01)),
      # fully connected, 7x7x64=3136 inputs 512 filters
      'wd1': tf.Variable(tf.truncated_normal([7*7*64, 512], stddev = 0.01)),
      # 512 inputs, 6 outputs (number of possible actions)
      'out': tf.Variable(tf.truncated_normal([512, self.n_actions], stddev = 0.01))
    }

    self.biases = {
      # 'bc1': tf.Variable(tf.fill([32],.1)),
      'bc1': tf.Variable(tf.constant(0.01, shape=[32])),
      # 'bc2': tf.Variable(tf.fill([64],.1)),
      'bc2': tf.Variable(tf.constant(0.01, shape=[64])),
      # 'bc3': tf.Variable(tf.fill([64],.1)),
      'bc3': tf.Variable(tf.constant(0.01, shape=[64])),
      # 'bd1': tf.Variable(tf.fill([512],.1)),
      'bd1': tf.Variable(tf.constant(0.01, shape=[512])),
      # 'out': tf.Variable(tf.fill([self.n_actions], .1)),
      'out': tf.Variable(tf.constant(0.01, shape=[self.n_actions])),
    }

    self.conv1 = DeepQNetwork.conv2d(self.x, self.weights['wc1'], self.biases['bc1'],4)
    assert t_shape(self.conv1) == (None, 20, 20, 32)

    self.conv2 = DeepQNetwork.conv2d(self.conv1, self.weights['wc2'], self.biases['bc2'], 2)
    assert t_shape(self.conv2) == (None, 9, 9, 64)

    self.conv3 = DeepQNetwork.conv2d(self.conv2, self.weights['wc3'], self.biases['bc3'], 1)
    assert t_shape(self.conv3) == (None, 7, 7, 64)

    fc1 = tf.reshape(self.conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
    self.fc1 = tf.nn.relu(fc1)

    # apply dropout
    # fc1 = tf.nn.dropout(fc1, C.net_dropout)

    assert t_shape(self.fc1) == (None, 512)

    self.out = tf.add(tf.matmul(self.fc1, self.weights['out']), self.biases['out'])
    assert t_shape(self.out) == (None, 6)

    # make the action mask to filter the relevant Q value for the taken action
    action_mask = tf.one_hot(self.actions_taken, depth=self.n_actions, on_value=1.0, off_value=0.0)
    assert t_shape(action_mask) == (self.batch_size,6)

    # y is already (None, 1)
    targets = tf.tile(self.y, [1, self.n_actions])
    assert t_shape(targets) == (None, 6)

    # gets the difference between target and prediction. Everything else is 0
    difference = action_mask * (targets - self.out)

    self.test = difference 
    self.loss = tf.nn.l2_loss(difference)
    self.loss_sum = tf.reduce_sum(self.loss)
    self.optimizer = tf.train.AdamOptimizer(C.net_learning_rate).minimize(self.loss)
    #self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum = C.net_rmsprop_momentum, epsilon=C.net_rmsprop_epsilon).minimize(self.loss)

    # start the session and init
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer()) 

    #### SAVER ##### 
    self.saver = tf.train.Saver()

    self.epoch = 1
    self.runs = 1
    self.trained_called = 0 # how many times have we trained?
    self.game_num = 0

    self.callback = None 

    # restore saved session if we need to
    if C.net_should_restore:
      print "Restoring previous session from path: {}".format(C.net_restore_path)
      self.saver.restore(self.sess, C.net_restore_path)
  
  @staticmethod
  def conv2d(x, W, b, strides):
    """
    Conv2D wrapper, with bias and relu activation
    """
    result = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    result = tf.nn.bias_add(result, b)
    return tf.nn.relu(result)

  def print_epoch(self,line):
    if self.runs % C.STEPS_PER_EPOCH == 0:
      print(line)

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

  def replay_memory_size(self):
    return len(self.replay_memory)

  def predict(self,state):
    """
      attributes:
        state - array of size (batch_size,84,84,4)
      returns:
        np array of size (batch_size,) 
        represents of the max Q-value for each example in the batch
    """
    assert state.shape == (32, 84, 84, 4)

    result = self.sess.run(self.out, feed_dict={self.x: state})
    assert result.shape == (32,6)

    ret = self.discount_factor * np.amax(result, axis=1)
    assert ret.shape == (32,)

    return ret

  def take_action(self,state):
    """
      attributes:
        state - DeepQNetworkState object
      returns:
        np array of a single number
        a single action number with the highest Q-value
    """
    assert state.screens.shape == (84,84,4)
    state = state.screens.reshape(1, 84, 84, 4)

    result = self.sess.run(self.out, feed_dict={self.x: state})
    assert result.shape == (1,6)

    return np.argmax(result[0,:]) # the first index is the batch

  def train_n_samples(self, transitions):
    """
      Params:
        transitions [list] - list of tuples representing the sequence (s, a, r, ns, d). s and ns are DeepQNetworkState objects. The length should always equal batch size.
    """
    assert len(transitions) == C.ai_batch_size
    self.trained_called += 1

    batch_size = C.ai_batch_size
    states = np.ndarray((batch_size,84,84,4))
    new_states = np.ndarray((batch_size,84,84,4))
    rewards = np.ndarray(batch_size) # shape (32,)
    actions = np.ndarray(batch_size) # shape (32,)

    for i in range(batch_size):
      s, a, r, ns, _ = transitions[i]
      states[i,:,:,:] = s.screens[:,:,:]
      new_states[i,:,:,:] = ns.screens[:,:,:]
      rewards[i] = r
      actions[i] = a

    # estimate the Q(s,a) for each batch
    target = rewards + self.predict(new_states)
    target = target[:,np.newaxis]
    assert target.shape == (32, 1)

    for i in range(batch_size):
      s,a,r,ns,d = transitions[i]
      # if terminal state make target the reward 
      if d:
        target[i] = r

    # train network with gradient descent and get loss
    _, loss_sum = self.sess.run([self.optimizer, self.loss_sum], feed_dict={self.x: states, 
                                                                            self.y: target, 
                                                                            self.actions_taken: actions})
    # get the q_values on validation set to get the mean q values
    if self.validation == False:
      self.validation_set = states
      self.validation = True

    # give statistics the loss_sum and qvalues 
    if self.callback:
      self.callback.on_train(loss_sum, self.trained_called)

  def save(self):
    path_with_epoch = "{}-{}.ckpt".format(C.net_save_path, self.epoch)
    print "Saving session to path: {}".format(path_with_epoch)
    self.saver.save(self.sess, path_with_epoch)
    print "epoch:{}, runs:{}, game_num:{}".format(self.epoch, self.runs, self.game_num)


