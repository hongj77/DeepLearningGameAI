import gym
from gym import wrappers
from gym import spaces
import constants as C

class OpenAIGym:
  """
  Wrapper class to interact with OpenAIGym Game

  attributes:
    env [gym.Environment] - environment of game
    game_name [string] - name of game being played 
    render [bool] - set to true if game should be visible during play 
    player_name [string] - OpenAI Gym account name. Either "hong" or "natasha".
    upload_name [int] - name of uploaded file (/tmp/game_name-version_num)
  """

  def __init__(self, game_name, render, player_name, version_num):
    self.env = gym.make(game_name)
    self.game_name = game_name
    self.upload_name = '/tmp/' + game_name + '-' + str(version_num)
    self.env = wrappers.Monitor(self.env, self.upload_name, force=True)
    self.render = render # True if we want to see the game during playing
    self.player = player_name

  def take_action(self, action):
    """
      Params:
        action [int] - A number in the range [0, action_space)

      Returns:
        observation [object] - environment-specific object. In most cases, screen pixels. i.e. (210, 160, 3) for breakout.
        reward [float] - scale varies between environments
        done [bool] - termination of episode
        info [dict] - diagnostics for debugging. Can't use in official agents. Seems useless in most cases.
    """
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
    gym.upload(self.upload_name, api_key=C.API_KEY[self.player])

