from network import DeepQNetwork
from network import DeepQNetworkState
import numpy as np
import random

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
    self.env = env

    # TODO: implement network
    self.network = DeepQNetwork()

    # hyperparameters
    self.learning_rate = .85
    self.future_discount = .99
    self.num_episodes = 100
    self.num_episode_length = 100
    self.batch_size = 32 #Google's DeepMind number 

  def train(self): 
    pass

  def test(self): 
    pass

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

  def play_nn(self): 
    epsilon = 1
    for g in range(self.num_episodes):
        print "starting game:", g
        state = self.env.reset()
        prepared_state = DeepQNetworkState.preprocess(state)
        network_state = DeepQNetworkState(prepared_state,np.zeros(prepared_state.shape),np.zeros(prepared_state.shape),np.zeros(prepared_state.shape))
        total_reward = 0
        while True:
            self.env.render_screen()
            
            #with small prob pick random action 
            if random.uniform(0,1) <= epsilon:
                action = int(np.floor(np.random.uniform(0,self.env.total_moves())))
                print(action)
            else:
                action = self.network.take_action(network_state)

            epsilon -= 1./1000000
            #action = np.argmax(np.random.randn(1,self.env.total_moves())*(1./(g+1)))
            new_state, reward, done, info = self.env.take_action(action)

            if done:
                break 
            
            new_network_state = DeepQNetworkState(DeepQNetworkState.preprocess(new_state), network_state.s0, network_state.s1, network_state.s2)

            self.network.insert_tuple_into_replay_memory((network_state,action,reward,new_network_state,done))

            # train cnn 
            if self.batch_size < self.network.replay_memory_size():
                batch = self.network.sample_random_replay_memory(self.batch_size)
                self.network.train_n_samples(batch)

            state = new_state
            network_state = new_network_state
            total_reward += reward 


