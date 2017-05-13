# ACKNOWLEDGEMENT - Modified from https://github.com/tambetm/simple_dqn/blob/master/src/statistics.py
import numpy as np
import tensorflow as tf
from tensorflow import contrib
import random
import scipy
from scipy import misc
import csv
import time 
import sys
import constants as C
import pdb

class Stats:
  """
  Keey running statistics on everything we want to plot.
  These statistics will only get plotted every C.STEPS_PER_PLOT
  """
  def __init__(self, network, game, mem):
    self.mem = mem
    self.network = network 
    self.game = game 
    self.network.callback = self
    self.csv_path = C.stats_csv_path 

    self.validation = False
    self.validation_set = None

    if self.csv_path != "":
      self.csv_file = open(self.csv_path, "wb")
      self.csv_writer = csv.writer(self.csv_file)

      self.csv_writer.writerow((
            "epoch",
            "steps",
            "average_reward_per_game",
            "average_q",
            "average_cost",
            "num_games_per_epoch",
            "epoch_max_reward",
            "epoch_min_reward",
          ))
      self.csv_file.flush()

    self.num_games_total = 0
    self.epoch = 0
    self.num_steps = 0
    self.game_rewards = 0 # running tally of rewards for the current game
    self.average_reward_per_game = 0 # running average
    self.average_cost = 0 # running average
    self.average_q = 0 # running average

    # these are on a per epoch basis
    self.epoch_max_reward = 0
    self.epoch_min_reward = 999999
    self.num_games_per_epoch = 0

  # call on step in game 
  def on_step(self, action, reward, terminal):
    self.game_rewards += reward
    self.num_steps += 1

    if terminal:
      self.num_games_total += 1
      self.num_games_per_epoch += 1
      self.epoch_max_reward = max(self.epoch_max_reward, self.game_rewards)
      self.epoch_min_reward = min(self.epoch_min_reward, self.game_rewards)
      self.average_reward_per_game += float(self.game_rewards - self.average_reward_per_game) / self.num_games_total
      self.game_rewards = 0 # reset current running reward

  # call every batch update
  def on_train(self, loss, runs):
    self.average_cost += float(loss - self.average_cost) / float(runs)
  
  def write(self, epoch):
    self.epoch = epoch
    print "Plotted Statistics at Epoch: {}".format(self.epoch)
    
    if not self.validation:
      self.validation = True
      states, actions, rewards, new_states, terminals = self.mem.getMinibatch()
      self.validation_set = states
      
    if self.validation:
      max_qvalues = self.network.predict(self.validation_set)
      self.average_q = np.mean(max_qvalues)
    else:
      self.average_q = 0

    if self.csv_path != "":
      self.csv_writer.writerow((
          self.epoch,
          self.num_steps,
          self.average_reward_per_game,
          self.average_q,
          self.average_cost,
          self.num_games_per_epoch,
          self.epoch_max_reward,
          self.epoch_min_reward,
        ))
      self.csv_file.flush()

    # reset all stats that are per epoch only
    self.epoch_max_reward = 0
    self.epoch_min_reward = 999999
    self.num_games_per_epoch = 0
    self.average_cost = 0
    self.network.trained_called = 0
    self.average_reward_per_game = 0
    self.num_games_total = 0
    

  def close(self):
    if self.csv_path:
      self.csv_file.close()



