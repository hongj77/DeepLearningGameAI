import gym
from gym import wrappers
from gym import spaces
import constants as C

API_KEY = 'sk_9Ft7yJrgT2M4k7y7Fe8A'

class OpenAIGym:
  """
  Wrapper class to interact with OpenAIGym Game

  attributes:
    env [gym.Environment] - environment of game
    render [bool] - set to true if game should be visible during play 
    game_name [string] - name of game being played 
    upload_name [int] - name of uploaded file (/tmp/game_name-version_num)
  """

  def __init__(self, game_name, render, version_num = 0):
    self.env = gym.make(game_name)
    self.game_name = game_name
    self.upload_name = '/tmp/' + game_name + '-' + str(version_num)
    self.env = wrappers.Monitor(self.env, self.upload_name, force=True)
    # True if we want to see the game during playing
    self.render = render

  def take_action(self, action):
    return self.env.step(action)

  def render_screen(self):
    if self.render:
      self.env.render()

  def reset(self):
    return self.env.reset()

  def close(self):
    self.env.close()

  def total_moves(self):
    return self.env.action_space.n

  def screen_space(self):
    return self.env.observation_space.n

  def upload_game(self):
    self.env.close()
    gym.upload(self.upload_name, api_key= API_KEY)

