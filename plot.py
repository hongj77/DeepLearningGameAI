# ACKNOWLEDGEMENT - Modified from https://github.com/tambetm/simple_dqn/blob/master/src/plot.py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import constants as C

# Change these
plot_figure_height = 10
plot_figure_width = 20
plot_stats = ["average_reward_per_game","average_q","average_cost", "num_games_per_epoch", "epoch_max_reward", "epoch_min_reward"]

csv_path = C.stats_csv_path
png_save_path = C.plot_png_path 
figure_height = plot_figure_height
figure_width = plot_figure_width 
fields = plot_stats

# field definitions for numpy
dtype = [
  ("epoch", "int"), 
  ("steps", "int"),
  ("average_reward_per_game", "float"),
  ("average_q", "float"),
  ("average_cost", "float"),
  ("num_games_per_epoch", "int"),
  ("epoch_max_reward", "float"),
  ("epoch_min_reward", "float")
]
data = np.loadtxt(csv_path, skiprows = 1, delimiter = ",", dtype = dtype)

# definitions for plot titles
labels = {
  "epoch": "Epoch", 
  "steps": "Number of Steps",
  "average_reward_per_game": "Average Reward Per Game",
  "average_q": "Average Q-value",
  "average_cost": "Average Loss",
  "num_games_per_epoch": "Number of Games Per Epoch",
  "epoch_max_reward": "Max Reward Per Epoch",
  "epoch_min_reward": "Min Reward Per Epoch"
}

# calculate number of subplots
nr_fields = len(fields)
cols = math.ceil(math.sqrt(nr_fields))
rows = math.ceil(nr_fields / float(cols))

plt.figure(figsize = (figure_width, figure_height))

# plot all fields
for i, field in enumerate(fields):
  plt.subplot(rows, cols, i + 1)

  plt.plot(data['epoch'], data[field])
  plt.ylabel(labels[field])
  plt.xlabel(labels['epoch'])
  plt.title(labels[field])

plt.tight_layout()

# if png_file argument given, save to file
if png_save_path != "":
  plt.savefig(png_save_path)
else:
  plt.show()