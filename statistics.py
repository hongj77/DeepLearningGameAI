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

class Stats:

  def __init__(self, network, game):
    self.network = network 
    self.game = game 

    self.network.callback = self

    self.csv_path = C.stats_csv_path 

    if self.csv_path != "":
      self.csv_file = open(self.csv_path, "wb")
      self.csv_writer = csv.writer(self.csv_file)

      self.csv_writer.writerow((
            "epoch",
            "steps",
            "nr_games",
            "average_reward",
            "total_train_steps",
            "meanq",
            "q_per_epoch",
            "meancost",
            "cost_per_epoch"
            "total_time",
            "epoch_time",
            "steps_per_second"
          ))
      self.csv_file.flush()

    self.start_time = time.clock()
    self.num_games = 0
    self.epoch = 0
    self.epoch_start_time = time.clock()
    self.num_steps = 0
    self.game_rewards = 0
    self.average_reward = 0
    self.average_cost = 0
    self.meanq = 0
    self.cost = 0
    self.q = 0

  #call on step in game 
  def on_step(self, action, reward, terminal):
    self.game_rewards += reward
    self.num_steps += 1

    if terminal:
      self.num_games += 1
      self.average_reward += float(self.game_rewards - self.average_reward) / self.num_games
      self.game_rewards = 0

  #call once cnn has trained (aka one epoch)
  def on_train(self, cost, qvalues):
    self.cost = cost
    self.epoch += 1
    self.epoch_start_time = time.clock()
    self.average_cost += (cost - self.average_cost) / float(self.epoch)
    self.meanq += self.meanq + (np.sum(qvalues)/ qvalues.shape[0])
    self.q = np.sum(qvalues)
    self.write()
  
  def write(self):
    current_time = time.clock()
    total_time = current_time - self.start_time
    epoch_time = current_time - self.epoch_start_time
    steps_per_second = self.num_steps / epoch_time

    if self.csv_path != "":
      self.csv_writer.writerow((
          self.epoch,
          self.num_steps,
          self.num_games,
          self.average_reward,
          self.num_steps,
          self.meanq,
          self.q,
          np.log(self.average_cost),
          np.log(self.cost),
          total_time,
          epoch_time,
          steps_per_second
        ))
      self.csv_file.flush()
    

  def close(self):
    if self.csv_path:
      self.csv_file.close()



