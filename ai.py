from network import DeepQNetwork
from network import DeepQNetworkState
import numpy as np
import random
from statistics import Stats 
import constants as C
import matplotlib.pyplot as plt
import pdb
import utils



class AI:
  """AI agent that we use to interact with the game environment. 

    attributes:
        env [gym.Environment] 
        Q [np.ndarray] - Q-function array Q[s,a] which indicates the quality of action a from state s, calculates max total reward 
        network [DeepQNetwork] - CNN that learns Q-function 
        learning_rate [float]
        futrue_discount [float]
        num_episodes [int]
        num_episode_length [int]
  """

  def __init__(self, env):
    self.env = env # custom OpenAIGym class
    self.learning_rate = C.ai_qtable_learning_rate
    self.future_discount = C.ai_qtable_future_discount
    self.num_episode_length = C.ai_qtable_num_episode_length
    self.epsilon = C.ai_init_epsilon 
    self.final_epsilon = C.ai_final_epsilon
    self.network = DeepQNetwork()
    self.stats = Stats(self.network, self.env)

  def play_qtable(self):    
    Q = np.zeros([self.env.screen_space(),self.env.total_moves()]);
    for g in range(self.num_episodes):
        state = self.env.reset()
        total_reward = 0
        self.env.render_screen()
        for _ in range(self.num_episode_length):
            #pick action w/ largest Q plus noise (more noise in beginning)
            action = np.argmax(Q[state,:] + np.random.randn(1,self.env.total_moves())*(1./(g+1)))
            #take a step in that direction 
            new_state, reward, done, info = self.env.take_action(action)
            #update Q table of current state,action
            Q[state,action] = Q[state,action] + self.learning_rate * (reward + self.future_discount * np.max(Q[new_state,:]) - Q[state,action])
            
            total_reward += reward 
            state = new_state

            if done:
                break

  def test_nn(self):
    print "TODO"

  def initialize_replay(self):
    print "Initializing ReplayMemory with count {}".format(C.ai_replay_mem_start_size)
    # until replay memory is initialized, play games
    while self.network.mem.count < C.ai_replay_mem_start_size:
      state = self.env.reset()
      print "Relpay Memory size: {}".format(self.network.mem.count)

      while True:
        self.env.render_screen()
        # format the current state to what we want
        prepared_state = utils.preprocess(state)

        # pick a random move and make the move
        action = random.randrange(self.env.total_moves())
        new_state, reward, done, info = self.env.take_action(action)

        # add to replay memory
        self.network.mem.add(action, reward, prepared_state, done)
        state = new_state

        if done:
          break
    print "Replay Memory initialization done!"

  def random_move(self):
    return random.randrange(self.env.total_moves())

  def train_nn(self): 
    num_steps = 0
    epoch = 0
    g = 0

    while epoch <= C.RUN_TILL_EPOCH:
      g += 1
      self.network.game_num = g
      print "Starting game: {}, total steps: {}, memory count: {}".format(g, num_steps, self.network.mem.count)

      state = self.env.reset()
      assert state.shape == (210,160,3) # only for breakout right now

      # for the first screen, initialize to something random
      prepared_state = utils.preprocess(state)
      for _ in range(C.net_num_screens):
        self.network.mem.add(self.random_move(), 0, prepared_state, False)

      # play until the AI loses or until the game completes
      while True:
        self.env.render_screen()
        num_steps += 1
        self.network.runs = num_steps

        # get the last four things from the replay memory
        state_with_history = self.network.mem.getLast()

        # take one step
        if random.random() < self.epsilon:
          action = self.random_move()
        else:
          action = self.network.take_action(state_with_history)

        # sanity check
        assert (action < self.env.total_moves() and action >= 0)

        # adjust epsilon for annealing
        if self.epsilon > self.final_epsilon:
          self.epsilon -= C.ai_epsilon_anneal_rate

        # make the move
        new_state, reward, done, info = self.env.take_action(action)
        
        # add to replay memory
        self.network.mem.add(action, reward, prepared_state, done)

        # train cnn every x number of runs
        if num_steps % C.net_train_rate == 0:
          transitions = self.network.mem.getMinibatch()
          self.network.train_n_samples(transitions)

        # aggregate stats on every training step
        self.stats.on_step(action, reward, done)

        # save network weights every epoch
        if C.net_should_save and num_steps % C.STEPS_PER_EPOCH == 0:
          self.network.save()

        # plot every epoch
        if num_steps % C.STEPS_PER_EPOCH == 0:
          epoch += 1
          self.network.epoch = epoch
          self.stats.write(epoch)

        # set variables for next loop
        prepared_state = utils.preprocess(new_state)

        # go to the next game if done
        if done:
          break


