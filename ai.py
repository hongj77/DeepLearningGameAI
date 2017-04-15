from network import DeepQNetwork
import numpy as np

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

    # temporary Q-table
    self.Q = np.zeros([self.env.screenSpace(),self.env.totalMoves()]);

    # hyperparameters
    self.learning_rate = .85
    self.future_discount = .99
    self.num_episodes = 1000
    self.num_episode_length = 100

  def train(self): 
    pass

  def test(self): 
    pass

  def play_qtable(self):    
    for g in range(self.num_episodes):
        state = self.env.reset()
        total_reward = 0
        self.env.renderScreen()
        for _ in range(self.num_episode_length):
            #pick action w/ largest Q plus noise (more noise in beginning)
            action = np.argmax(self.Q[state,:] + np.random.randn(1,self.env.totalMoves())*(1./(g+1)))
            #take a step in that direction 
            new_state, reward, done, info = self.env.take_action(action)
            #update Q table of current state,action
            self.Q[state,action] = self.Q[state,action] + self.learning_rate * (reward + self.future_discount * np.max(self.Q[new_state,:]) - self.Q[state,action])
            
            total_reward += reward 
            state = new_state

            if done:
                break

        if g == self.num_episodes - 1:
            print(total_reward)

  def play_nn(self): 
    pass
