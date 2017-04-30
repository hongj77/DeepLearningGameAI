import numpy as np 
from ai import AI
from game import OpenAIGym
from statistics import Stats
import constants as C

if __name__=="__main__":
  game_name = "Breakout-v0"
  render_to_screen = True
  player_name = "hong" # hong or natasha
  version_num = 0

  env = OpenAIGym(game_name, render_to_screen, player_name, version_num)
  ai = AI(env)
  ai.play_nn()
  env.upload_game()