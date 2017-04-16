import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow import contrib
import random
import scipy
from scipy import misc

class DeepQNetwork:
  """
    attributes:
      replayMemory - [(s,a,r,s')] array of 4-tuple representing replay memory 
  """
  
  def __init__(self): 
  # initializes the layers of the CNN
    self.replay_memory = []
    learning_rate = .05

    height = 84
    width = 84
    num_screens = 4
    # number of possible outputs of the network
    n_actions = 6

    self.x = tf.placeholder(tf.float32, [None, height*width*num_screens])
    self.y = tf.placeholder(tf.float32, [None, 1])
    self.a = tf.placeholder(tf.int32, [None,1])
    # reshape to a 4D tensor
    self.x_tensor = tf.reshape(self.x, [-1, height, width, num_screens])

    self.weights = {
      # 8x8 filter, 4 inputs 32 outputs
      'wc1': tf.Variable(tf.random_normal([8,8,4,32])),
      # 4x4 filter, 32 inputs 64 outputs
      'wc2': tf.Variable(tf.random_normal([4,4,32,64])),
      # 3x3 filter, 64 inputs 64 outputs
      'wc3': tf.Variable(tf.random_normal([3,3,64,64])),
      # fully connected, 7x7x64 inputs 512 outputs
      'wd1': tf.Variable(tf.random_normal([7*7*64, 512])),
      # 512 inputs, 18 outputs (number of possible actions)
      'out': tf.Variable(tf.random_normal([512, n_actions]))
    }

    self.biases = {
      'bc1': tf.Variable(tf.zeros([32])),
      'bc2': tf.Variable(tf.zeros([64])),
      'bc3': tf.Variable(tf.zeros([64])),
      'bd1': tf.Variable(tf.zeros([512])),
      'out': tf.Variable(tf.zeros([n_actions])),
    }

    self.conv1 = DeepQNetwork.conv2d(self.x_tensor, self.weights['wc1'], self.biases['bc1'],4)
    self.conv2 = DeepQNetwork.conv2d(self.conv1, self.weights['wc2'], self.biases['bc2'],2)
    self.conv3 = DeepQNetwork.conv2d(self.conv2, self.weights['wc3'], self.biases['bc3'])
    fc1 = tf.reshape(self.conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
    self.fc1 = tf.nn.relu(fc1)
    self.out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
    #self.preds = tf.gather(self.out[:,:],self.a)
    self.preds = tf.gather(self.out, self.a)
    print(self.preds.shape)
    self.loss = 1./2 * tf.reduce_mean(tf.square(self.preds-self.y))
    #self.loss = tf.losses.log_loss(self.y,self.preds)
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer()) 

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
    
  def _createNetwork(): pass

  # trains the network using batches of replay memory
  def train(self, transition):
    # sample data filled with all 0's representing a Grayscale game screen
    s, a, r, ns, d = transition 
    ns = ns.screens.reshape(1,84*84*4)
    s = s.screens.reshape(1,84*84*4)
    target = np.zeros((1,1))
    action = np.zeros((1,1))
    action[0] = a
    if d:
        target[i] = r
    else:
      target[0] = self.predict(ns) + r
    self.sess.run(self.optimizer, feed_dict={self.x: s, self.y: target, self.a: action})
    #print(self.sess.run(self.loss, feed_dict={self.x: s, self.y: [target]}))

  def train_n_samples(self,transitions):
    batch_size = len(transitions)
    states = np.ndarray((84,84,4,batch_size))
    new_states = np.ndarray((84,84,4,batch_size))
    rewards = np.ndarray((batch_size))
    target = np.ndarray((batch_size,1))
    actions = np.ndarray((batch_size,1))
    for i in range(batch_size):
      s, a, r, ns, _ = transitions[i]
      ns = ns.screens[:,:,:]
      s = s.screens[:,:,:]
      states[:,:,:,i] = s
      new_states[:,:,:,i] = ns 
      rewards[i] = r
      actions[i] = a
      
    states = states.reshape(batch_size,84*84*4)
    new_states = states.reshape(batch_size,84*84*4)
    target = self.predict(new_states) + rewards 

    for i in range(batch_size):
      _,_,_,_,d = transitions[i]
      if d:
        target[i] = r

    self.sess.run(self.optimizer, feed_dict={self.x: states, self.y: target[:,np.newaxis], self.a: actions})
    print(self.sess.run(self.preds, feed_dict={self.x: states, self.y: target[:,np.newaxis], self.a: actions}))

  def predict(self,state):
    result = self.sess.run(self.out, feed_dict={self.x: state})
    return np.amax(result[:,:], axis = 1)

  def take_action(self,state):
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
    sg = scipy.misc.imresize(sg,(84,84),interp="nearest")
    return sg[:,:,np.newaxis]

  @staticmethod
  def convert_to_grayscale(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])



