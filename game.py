import gym
from gym import wrappers
from gym import spaces

class OpenAIGym:
  def __init__(self, game_name, render):
    self.env = gym.make(game_name)
    self.env = wrappers.Monitor(self.env, '/tmp/frozenlake', force=True)
    # True if we want to see the game during playing
    self.render = render

  def take_action(self, action):
    return self.env.step(action)

  def renderScreen(self):
    if self.render:
      self.env.render()

  def reset(self):
    return self.env.reset()

  def totalMoves(self):
    return self.env.action_space.n

  def screenSpace(self):
    return self.env.observation_space.n

