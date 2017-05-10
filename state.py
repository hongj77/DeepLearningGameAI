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