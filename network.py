import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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

class DeepQNetwork:
  def __init__(self, batch_size, save_cur_sess = False, save_path = "", restore_path = ""): 
    self.replay_memory = [] # [(s,a,r,s',d)] array of 4-tuple representing replay memory 
    self.batch_size = batch_size
    self.height = C.net_height
    self.width = C.net_width 
    self.num_screens = C.net_num_screens
    self.n_actions = C.net_n_actions
    self.discount_factor = C.net_discount_factor

    # a single sample of a game state that we hold on to for mean q-values over time
    self.validation_set = None

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
    difference = action_mask * (self.out - targets)
    self.test = difference 

    self.loss = tf.nn.l2_loss(difference)
    self.loss_sum = tf.reduce_sum(self.loss)
    self.optimizer = tf.train.AdamOptimizer(C.net_learning_rate).minimize(self.loss)
    #self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum = C.net_rmsprop_momentum, epsilon=C.net_rmsprop_epsilon).minimize(self.loss)

    # start the session and init
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer()) 

    #### SAVER ##### 
    self.save_path = save_path
    self.save_cur_sess = save_cur_sess
    self.restore_path= restore_path
    self.runs = 0
    self.runs_till_save = C.net_runs_till_save
    self.saver = tf.train.Saver()
    
    self.callback = None 
    # restore saved session 
    if restore_path != "":
        self.saver.restore(self.sess, restore_path)
  
  
  @staticmethod
  def conv2d(x, W, b, strides):
    """
    Conv2D wrapper, with bias and relu activation
    """
    result = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    result = tf.nn.bias_add(result, b)
    return tf.nn.relu(result)

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
    
  # trains the network using batches of replay memory
  def train(self, transition):

    # sample data filled with all 0's representing a Grayscale game screen
    s, a, r, ns, d = transition 
    target = np.zeros((1,1))

    # if the game is done, then the reward is the actual target reward
    if d:
      target[i] = r
    else:
      target[0] = self.predict(ns) + r

    self.sess.run(self.optimizer, feed_dict={self.x: s, self.y: target})
    loss_sum = self.sess.run(self.loss_sum, feed_dict={self.x: states, 
                                                       self.y: target[:,np.newaxis], 
                                                       self.actions_taken: actions})
    qvalue = self.sess.run(self.out, feed_dict={self.x: state})

     #save every 1000 runs 
    if self.save_cur_sess and self.runs % self.runs_till_save == 0:
      self.saver.save(self.sess, self.save_path)

    if self.callback:
      self.callback.on_train(loss_sum,qvalue, self.runs)
    self.runs += 1


  def train_n_samples(self, transitions):
    """
      Params:
        transitions [list] - list of tuples representing the sequence (s, a, r, ns, d). s and ns are DeepQNetworkState objects. The length should always equal batch size.
    """
    assert len(transitions) == C.ai_batch_size

    batch_size = C.ai_batch_size
    states = np.ndarray((batch_size,84,84,4))
    new_states = np.ndarray((batch_size,84,84,4))
    rewards = np.ndarray(batch_size) # shape (32,)
    actions = np.ndarray(batch_size) # shape (32,)

    for i in range(batch_size):
      s, a, r, ns, _ = transitions[i]
      s = s.screens[:,:,:] # may be unneccessary?
      ns = ns.screens[:,:,:]
      states[i,:,:,:] = s
      new_states[i,:,:,:] = ns 
      rewards[i] = r
      actions[i] = a

    # estimate the Q(s,a) for each batch
    target = rewards + self.predict(new_states)
    target = target[:,np.newaxis]
    assert target.shape == (32, 1)

    for i in range(batch_size):
      _,_,_,_,d = transitions[i]
      # if terminal state make target the reward 
      if d:
        target[i] = r

    # train network with gradient descent and get loss
    _, loss_sum = self.sess.run([self.optimizer, self.loss_sum], feed_dict={self.x: states, 
                                                                        self.y: target, 
                                                                        self.actions_taken: actions})
    print "loss sum: {}".format(loss_sum)
    
    # get the q_values on validation set to get the mean q values
    if self.validation_set == None:
      self.validation_set = states

    # qvalues = self.sess.run(self.out, feed_dict={self.x: self.validation_set})
    qvalues = self.predict(self.validation_set)
    assert qvalues.shape == (32,)

    # save every runs_till_save number of runs 
    if self.save_cur_sess and self.runs % self.runs_till_save == 0:
      self.saver.save(self.sess, self.save_path)

    self.runs += 1

    if self.callback:
      # give statistics the loss_sum and qvalues 
      self.callback.on_train(loss_sum, qvalues, self.runs)


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
    self.screens = np.concatenate((s0,s1,s2,s3), axis=2)

  @staticmethod
  def preprocess(state):
    """
    Params:
      state - takes in (h,w,3) 

    Returns:
      preprocessed state that resizes the image and grayscales it
      returns an image with shape (C.net_height, C.net_width, 1)
    """
    sg = DeepQNetworkState.convert_to_grayscale(state)
    #Google used bilinear 
    sg = scipy.misc.imresize(sg,(110,84),interp="bilinear")

    #crop to 84 x 84 which is most of game play screen 
    sg = sg[(110-84):,:]

    assert sg.shape == (C.net_height, C.net_width)
    return sg[:,:,np.newaxis]

  @staticmethod
  def convert_to_grayscale(image):
    """
    Params:
      state - shape (h,w,3) the 3 represents r,g,b
    Returns:
      grayscaled image of (h,w,3)
    """
    grayed = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return grayed



