from network import DeepQNetwork
from network import DeepQNetworkState
import numpy as np
import random
from statistics import Stats 
import constants as C
import matplotlib.pyplot as plt

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
    self.learning_rate = C.ai_qtable_learning_rate
    self.future_discount = C.ai_qtable_future_discount
    self.num_episodes = C.ai_num_episodes
    self.num_episode_length = C.ai_qtable_num_episode_length
    self.batch_size = C.ai_batch_size #Google's DeepMind number 
    self.epsilon = C.ai_init_epsilon 
    self.final_epsilon = C.ai_final_epsilon
    self.network = DeepQNetwork(self.batch_size, save_cur_sess = C.net_should_save, save_path = C.net_save_path, restore_path = C.net_restore_path)


    self.stats = Stats(self.network,self.env)

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
    epsilon = self.epsilon 

    for g in range(self.num_episodes):
        print "===============================starting game:", g
        state = self.env.reset()
        prepared_state = DeepQNetworkState.preprocess(state)
        network_state = DeepQNetworkState(prepared_state,np.zeros(prepared_state.shape),np.zeros(prepared_state.shape),np.zeros(prepared_state.shape))
        total_reward = 0
        while True:
            
            self.env.render_screen()
            
            #with small prob pick random action 
            uniform_n = np.random.uniform(0,1)
            if uniform_n <= epsilon:
                action = int(np.floor(np.random.uniform(0,self.env.total_moves())))
            else:
                action = self.network.take_action(network_state)

            if epsilon > self.final_epsilon and C.ai_replay_mem_start_size < self.network.replay_memory_size():
                epsilon -= 1./10000
            
            new_state, reward, done, info = self.env.take_action(action)
            
            new_network_state = DeepQNetworkState(DeepQNetworkState.preprocess(new_state), network_state.s0, network_state.s1, network_state.s2)
            self.network.insert_tuple_into_replay_memory((network_state,action,reward,new_network_state,done))

            """
            if self.network.replay_memory_size() == 20:
                plt.imshow((new_network_state.s0).reshape(84,84))
                plt.show()
            """

            # train cnn 
            if C.ai_replay_mem_start_size < self.network.replay_memory_size():
                batch = self.network.sample_random_replay_memory(self.batch_size)
                self.network.train_n_samples(batch)

            state = new_state
            network_state = new_network_state
            self.stats.on_step(action, reward, done)

            if done:
                break 


