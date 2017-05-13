from network import DeepQNetwork
from state import DeepQNetworkState
from deepnetwork import DeepQNetworkNeon
from replay import Replay
from state_buffer import StateBuffer
from replay_memory import ReplayMemory

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
    # self.network = DeepQNetwork()
    self.neon = DeepQNetworkNeon()
    # self.mem = Replay()
    self.mem = ReplayMemory()
    self.stats = Stats(self.neon, self.env, self.mem)
    self.buf = StateBuffer()

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

  def random_move(self):
    return random.randrange(self.env.total_moves())

  def test_nn(self):
    print "TODO"

  def restart_random(self):
    state = self.env.reset()
    for _ in range(random.randint(C.net_num_screens, C.random_starts) + 1):
      screen, reward, done, info = self.env.take_action(0)
      if done:
        state = self.env.reset()
      self.buf.add(screen)

  def train_nn(self): 
    num_steps = 0
    epoch = 0
    g = 0

    while epoch < C.RUN_TILL_EPOCH:
      g += 1
      self.neon.game_num = g
      print "Starting game: {}, total_steps: {}, memory size: {}".format(g, num_steps, self.mem.count)
      # restart random
      restart_random()
      # play until the AI loses or until the game completes
      while True:
        self.env.render_screen()
        num_steps += 1
        self.neon.runs = num_steps

        # get screen
        screen = self.buf.getStateMinibatch()
        
        # taking a step
        if random.random() < self.epsilon:
          action = self.random_move()
        else:
          # action = self.network.take_action(network_state)
          # screen = network_state.screens_neon()
          # pdb.set_trace()
          qvals = self.neon.predict(screen)
          action = np.argmax(qvals[0])

        # reduce epsilon by annealing rate
        if self.epsilon > self.final_epsilon and C.ai_replay_mem_start_size < self.mem.count:
          self.epsilon -= C.ai_epsilon_anneal_rate 
        
        # sanity check
        assert (action < self.env.total_moves() and action >= 0)

        # take an action
        new_state, reward, done, info = self.env.take_action(action)

        self.buf.add(new_state)

        # go to the next game if done
        if done:
          break

        # make new state and put the tuple in memory
        # new_network_state = DeepQNetworkState(new_state, network_state.s0, network_state.s1, network_state.s2)
        # memory = (network_state, action, reward, new_network_state, done)
        # self.mem.insert_tuple_into_replay_memory(memory)
        self.mem.add(action, reward, new_state, done)

        if C.target_steps and num_steps % C.target_steps == 0:
          self.neon.update_target_network()

        # train cnn 
        if C.ai_replay_mem_start_size < self.mem.count:
          # only train every x number of runs
          if num_steps % C.net_train_rate == 0:
            # batch = self.mem.sample_random_replay_memory(C.ai_batch_size)
            # self.network.train_n_samples(batch)
            # minibatch = self.mem.get_minibatch()
            minibatch = self.mem.getMinibatch()
            self.neon.train(minibatch, epoch)

        # aggregate stats on every training step
        self.stats.on_step(action, reward, done)

        # plot every epoch
        if num_steps % C.STEPS_PER_EPOCH == 0:
          epoch += 1
          self.neon.epoch = epoch
          self.stats.write(epoch)

        # save every epoch
        if C.net_should_save and num_steps % C.STEPS_PER_EPOCH == 0:
          self.neon.save()

        # set variables for next loop
        # network_state = new_network_state
