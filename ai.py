from network import DeepQNetwork
from network import DeepQNetworkState
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
    self.num_episodes = C.ai_num_episodes
    self.num_episode_length = C.ai_qtable_num_episode_length
    self.epsilon = C.ai_init_epsilon 
    self.final_epsilon = C.ai_final_epsilon
    self.network = DeepQNetwork(batch_size=C.ai_batch_size, 
                                save_cur_sess = C.net_should_save, 
                                save_path = C.net_save_path, 
                                restore_path = C.net_restore_path)
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

  def play_nn(self, training=True): 
    epsilon = self.epsilon 

    for g in range(self.num_episodes):
        print "===============================starting game:", g

        state = self.env.reset()

        assert state.shape == (210,160,3) # only for breakout right now

        prepared_state = DeepQNetworkState.preprocess(state)

        assert prepared_state.shape == (C.net_height, C.net_width, 1)

        # setting s1 - s3 to be a black image
        network_state = DeepQNetworkState(prepared_state, 
            np.zeros(prepared_state.shape),
            np.zeros(prepared_state.shape),
            np.zeros(prepared_state.shape))

        assert network_state.screens.shape == (C.net_height, C.net_width, 4)

        total_reward = 0

        # play until the AI loses or until the game completes
        num_steps = 0
        while True:
            num_steps += 1
            self.env.render_screen()

            # with small prob pick random action 
            uniform_n = np.random.uniform(0,1)
            if uniform_n <= epsilon:
                if num_steps % 10 == 0:
                    print "{}<={} took random action!!".format(uniform_n, epsilon)
                action = int(np.floor(np.random.uniform(0,self.env.total_moves())))

            else:
                # pick the action that the neural network suggests
                if num_steps % 10 == 0:
                    print "{}>{} took neural net action.....".format(uniform_n, epsilon)
                action = self.network.take_action(network_state)

            assert (action < self.env.total_moves() and action >= 0)

            if epsilon > self.final_epsilon and C.ai_replay_mem_start_size < self.network.replay_memory_size():
                epsilon -= C.ai_epsilon_anneal_rate 

            new_state, reward, done, info = self.env.take_action(action)

            if training:
                new_network_state = DeepQNetworkState(
                    DeepQNetworkState.preprocess(new_state), 
                    network_state.s0, 
                    network_state.s1, 
                    network_state.s2)

                self.network.insert_tuple_into_replay_memory((network_state,
                                                                action,
                                                                reward,
                                                                new_network_state,
                                                                done))

                """
                if self.network.replay_memory_size() == 20:
                    plt.imshow((new_network_state.s0).reshape(84,84))
                    plt.show()
                """

                # train cnn 
                if C.ai_replay_mem_start_size < self.network.replay_memory_size():
                    print "================TRAINING======================"
                    print "replay memory size: {}".format(self.network.replay_memory_size())

                    batch = self.network.sample_random_replay_memory(C.ai_batch_size)
                    self.network.train_n_samples(batch)

                state = new_state
                network_state = new_network_state
                self.stats.on_step(action, reward, done)

            if done:
                break 


