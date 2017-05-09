#!/usr/bin/env python

import numpy as np 
from ai import AI
from game import OpenAIGym
from statistics import Stats
import constants as C

if __name__=="__main__":
  game_name = "Breakout-v0"
  version_num = 0
  player_name = "hong" # hong or natasha
  training = not C.net_should_restore # if we are restoring from a file, we are testing and not training
  render_to_screen = True

  env = OpenAIGym(game_name, render_to_screen, player_name, version_num)
  ai = AI(env)

  if training:
    print "Training AI!"
  else:
    print "Testing AI with file: {}".format(C.net_restore_path)

  ai.play_nn(training)
  env.upload_game()