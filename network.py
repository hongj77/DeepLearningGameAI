import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow import contrib
import random
import scipy
from scipy import misc
import constants as C

class DeepQNetwork:
  """
    attributes:
      replayMemory - [(s,a,r,s')] array of 4-tuple representing replay memory 
  """
  
  def __init__(self, batch_size, save_cur_sess = False, save_path = "", restore_path = ""): 
  # initializes the layers of the CNN
    self.replay_memory = []
    learning_rate = C.net_learning_rate

    self.batch_size = batch_size
    self.height = C.net_height
    self.width = C.net_width 
    self.num_screens = C.net_num_screens
    # number of possible outputs of the network
    self.n_actions = C.net_n_actions
    self.discount_factor = C.net_discount_factor

    #PLACEHOLDERS 
    self.x = tf.placeholder(tf.float32, [None, self.height*self.width*self.num_screens])
    self.y = tf.placeholder(tf.float32, [None, 1])
    # expects shape (?, ) in ranges (0..5)
    self.actions_taken = tf.placeholder(tf.int32, [self.batch_size])

    # input into the network with shape (batch_size, 84,84,4)
    self.x_tensor = tf.reshape(self.x, [-1, self.height, self.width, self.num_screens])

    self.weights = {
      # 8x8 filter, 4 inputs 32 outputs
      'wc1': tf.Variable(tf.random_normal([8,8,4,32], stddev = 0.2)),
      # 4x4 filter, 32 inputs 64 outputs
      'wc2': tf.Variable(tf.random_normal([4,4,32,64], stddev = 0.2)),
      # 3x3 filter, 64 inputs 64 outputs
      'wc3': tf.Variable(tf.random_normal([3,3,64,64], stddev = 0.2)),
      # fully connected, 7x7x64 inputs 512 outputs
      'wd1': tf.Variable(tf.random_normal([7*7*64, 512], stddev = 0.2)),
      # 512 inputs, 18 outputs (number of possible actions)
      'out': tf.Variable(tf.random_normal([512, self.n_actions],stddev = 0.2))
    }

    self.biases = {
      'bc1': tf.Variable(tf.fill([32],.1)),
      'bc2': tf.Variable(tf.fill([64],.1)),
      'bc3': tf.Variable(tf.fill([64],.1)),
      'bd1': tf.Variable(tf.fill([512],.1)),
      'out': tf.Variable(tf.fill([self.n_actions], .1)),
    }

    self.conv1 = DeepQNetwork.conv2d(self.x_tensor, self.weights['wc1'], self.biases['bc1'],4)
    self.conv2 = DeepQNetwork.conv2d(self.conv1, self.weights['wc2'], self.biases['bc2'],2)
    self.conv3 = DeepQNetwork.conv2d(self.conv2, self.weights['wc3'], self.biases['bc3'])
    fc1 = tf.reshape(self.conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
    self.fc1 = tf.nn.relu(fc1)
    self.out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

    action_mask = tf.one_hot(
        self.actions_taken, depth=self.n_actions, on_value=1.0, off_value=0.0)

    targets = tf.tile(self.y, [1, self.n_actions])

    # gets the difference between target and prediction. Everything else is 0
    difference = action_mask * (self.y - self.out)
    self.test = difference 

    self.loss = tf.nn.l2_loss(difference)
    self.loss_sum = tf.reduce_sum(self.loss)
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

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
    #restore saved session 
    if restore_path != "":
        self.saver.restore(self.sess, restore_path)
  
  # create the conv_net model
  @staticmethod
  def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

  def insert_tuple_into_replay_memory(self, mem_tuple):
    assert len(mem_tuple) == 5
    assert type(mem_tuple) == tuple
    self.replay_memory.append(mem_tuple)

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
    ns = ns.screens.reshape(1,84*84*4)
    s = s.screens.reshape(1,84*84*4)
    target = np.zeros((1,1))
    if d:
        target[i] = r
    else:
      target[0] = self.predict(ns) + r
    self.sess.run(self.optimizer, feed_dict={self.x: s, self.y: target})
    loss_sum = self.sess.run(self.loss_sum, feed_dict={self.x: states, self.y: target[:,np.newaxis], self.actions_taken: actions})
    qvalue = self.sess.run(self.out, feed_dict={self.x: state})

     #save every 1000 runs 
    if self.save_cur_sess and self.runs % self.runs_till_save == 0:
      self.saver.save(self.sess, self.save_path)

    if self.callback:
      self.callback.on_train(loss_sum,qvalue)
    self.runs += 1

  def train_n_samples(self,transitions):
    batch_size = len(transitions)
    states = np.ndarray((batch_size,84,84,4))
    new_states = np.ndarray((batch_size,84,84,4))
    actions = np.ndarray(batch_size) # shape (?,)
    rewards = np.ndarray(batch_size)
    target = np.ndarray((batch_size,1))

    for i in range(batch_size):
      s, a, r, ns, _ = transitions[i]
      s = s.screens[:,:,:]
      ns = ns.screens[:,:,:]
      states[i,:,:,:] = s
      new_states[i,:,:,:] = ns 
      rewards[i] = r
      actions[i] = a
      
    states = states.reshape(batch_size,84*84*4)
    new_states = states.reshape(batch_size,84*84*4)

    target = self.predict(new_states) + rewards 
    assert target.shape == (batch_size, )

    for i in range(batch_size):
      _,_,_,_,d = transitions[i]
      #if terminal state make target the reward 
      if d:
        target[i] = r

    self.sess.run(self.optimizer, feed_dict={self.x: states, self.y: target[:,np.newaxis], self.actions_taken: actions})
    
    loss_sum = self.sess.run(self.loss_sum, feed_dict={self.x: states, self.y: target[:,np.newaxis], self.actions_taken: actions})
    print(loss_sum)
    
    qvalues = self.sess.run(self.out, feed_dict={self.x: states})

    #save every runs_till_save number of runs 
    if self.save_cur_sess and self.runs % self.runs_till_save == 0:
      self.saver.save(self.sess, self.save_path)

    self.runs += 1

    if self.callback:
      #give statistics the loss_sum and qvalues 
      self.callback.on_train(loss_sum,qvalues)


  def predict(self,state):
    """
      attributes:
        state - array of size (batch_size, 84x84x4)
      returns:
        np array of size (batch_size,) 
        represents of the max Q-value for each example in the batch
    """
    result = self.sess.run(self.out, feed_dict={self.x: state})
    ret = self.discount_factor * np.amax(result[:,:], axis = 1)
    return ret

  def take_action(self,state):
    """
      attributes:
        state - DeepQNetworkState object
      returns:
        np array of a single number
        a single action number with the highest Q-value
    """
    state = state.screens.reshape(1,84*84*4)
    result = self.sess.run(self.out, feed_dict={self.x: state})
    return np.argmax(result[0,:])

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
  def preprocess(state):
    sg = DeepQNetworkState.convert_to_grayscale(state)
    #Google used bilinear 
    sg = scipy.misc.imresize(sg,(84,84),interp="bilinear")
    return sg[:,:,np.newaxis]

  @staticmethod
  def convert_to_grayscale(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])



