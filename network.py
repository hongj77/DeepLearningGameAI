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

def test(self):

    # TEMP
    batch_size = 10 

    def conv2d(x, W, b, strides=1):
      # Conv2D wrapper, with bias and relu activation
      x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
      x = tf.nn.bias_add(x, b)
      return tf.nn.relu(x)

    height = 84
    width = 84
    num_screens = 4
    # number of possible outputs of the network
    n_actions = 18

    # 84 x 84 x 4 = 28224
    x = tf.placeholder(tf.float32, [None, height*width*num_screens])

    # 18
    y = tf.placeholder(tf.float32, [None, n_actions])

    # sample data filled with all 0's representing a Grayscale game screen
    sample_data = np.zeros([height, width, num_screens], dtype=np.float32)
    print(sample_data.shape)

    # reshape to a 4D tensor
    x_tensor = tf.reshape(sample_data, [-1, height, width, num_screens])
    print(x_tensor.shape)

    weights = {
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

    biases = {
      'bc1': tf.Variable(tf.random_normal([32])),
      'bc2': tf.Variable(tf.random_normal([64])),
      'bc3': tf.Variable(tf.random_normal([64])),
      'bd1': tf.Variable(tf.random_normal([512])),
      'out': tf.Variable(tf.random_normal([n_actions])),
    }

    # create the conv_net model
    conv1 = conv2d(x_tensor, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

    # reshape and add bias + relu
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    print out

    # # loss 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

    # loss = tf.reduce_sum(1.0)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()

    print "accuracy"
    print accuracy
    print "optimizer"
    print optimizer

    with tf.Session() as session:
      result = session.run(slice, feed_dict={image: raw_image_data})
      print(result.shape)

    # # evaluate model
    # correct_pred = tf.equal()

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





