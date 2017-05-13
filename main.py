#!/usr/bin/env python

import numpy as np 
from ai import AI
from game import OpenAIGym
from statistics import Stats
import constants as C

if __name__=="__main__":
  game_name = "DemonAttack-v0"
  version_num = 0
  player_name = "hong" # hong or natasha
  training = not C.net_should_restore # if we are restoring from a file, we are testing and not training
  render_to_screen = True

  env = OpenAIGym(game_name, render_to_screen, player_name, version_num)
  ai = AI(env)

  if training:
    print "="*10
    print "Training AI for {} epochs".format(C.RUN_TILL_EPOCH)
    print "="*10
    ai.train_nn()
  else:
    print "="*10
    print "Testing AI with file: {}".format(C.net_restore_path)
    print "="*10
    ai.test_nn()

  if C.UPLOAD:
    print "="*10
    print "Uploading game to OpenAI"
    print "="*10  
    env.upload_game()