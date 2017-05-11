# from network import DeepQNetwork
from network import DeepQNetworkState
from deepnetwork import DeepQNetwork2

import numpy as np
import random
from statistics import Stats 
import constants as C
import matplotlib.pyplot as plt
import pdb



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
    self.network = DeepQNetwork2(6,None)
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

  def play_nn(self, training): 
    num_steps = 0
    epoch = 0
    g = 0

    while epoch <= C.RUN_TILL_EPOCH:
      g += 1 # num games
      self.network.game_num = g
      print "Starting game: {}, total_steps: {}".format(g, num_steps)

      state = self.env.reset()
      assert state.shape == (210,160,3) # only for breakout right now

      prepared_state = DeepQNetworkState.preprocess(state)
      assert prepared_state.shape == (C.net_height, C.net_width, 1)

      # setting s1 - s3 to be a black image
      network_state = DeepQNetworkState(prepared_state, 
                                        np.zeros(prepared_state.shape),
                                        np.zeros(prepared_state.shape),
                                        np.zeros(prepared_state.shape))
      assert network_state.screens().shape == (C.net_height, C.net_width, 4)

      # play until the AI loses or until the game completes
      while True:
        self.env.render_screen()
        num_steps += 1

        self.network.runs = num_steps
        self.network.epoch = epoch

        # taking a step
        if training:
          # training so use epsilon sometimes
          if random.random() < self.epsilon:
            action = random.randrange(self.env.total_moves())
          else:
            action = self.network.take_action(network_state)

          if self.epsilon > self.final_epsilon and C.ai_replay_mem_start_size < self.network.replay_memory_size():
            self.epsilon -= C.ai_epsilon_anneal_rate 
        else:
          # testing so only use NN
          action = self.network.take_action(network_state)

        # sanity check
        assert (action < self.env.total_moves() and action >= 0)

        new_state, reward, done, info = self.env.take_action(action)
        new_network_state = DeepQNetworkState(DeepQNetworkState.preprocess(new_state), 
                                              network_state.s0, 
                                              network_state.s1, 
                                              network_state.s2)
        if training:
          # (n,a,r,ns,d) like in the paper
          memory = (network_state, action, reward, new_network_state, done)
          self.network.insert_tuple_into_replay_memory(memory)

          # train cnn 
          if C.ai_replay_mem_start_size < self.network.replay_memory_size():
            # only train every x number of runs
            if num_steps % C.net_train_rate == 0:
              batch = self.network.sample_random_replay_memory(C.ai_batch_size)
              self.network.train_n_samples(batch)

          # aggregate stats on every training step
          self.stats.on_step(action, reward, done)

          # save every runs_till_save number of runs if we need to
          if C.net_should_save and num_steps % C.STEPS_PER_EPOCH == 0:
            self.network.save()

        # plot every epoch
        if num_steps % C.STEPS_PER_EPOCH == 0:
          # only plot if training
          if training:
            self.stats.write(epoch)
          epoch += 1

        # set variables for next loop
        network_state = new_network_state

        # go to the next game if done
        if done:
          break


if __name__=="__main__":
  print 1

