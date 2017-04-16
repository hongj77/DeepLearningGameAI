import numpy as np 
from ai import AI
from game import OpenAIGym

if __name__=="__main__":
  env = OpenAIGym("Breakout-v0", True)
  ai = AI(env)
  ai.play_nn()
  env.upload_game()