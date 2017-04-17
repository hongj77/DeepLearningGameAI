# ACKNOWLEDGEMENT - Modified from https://github.com/tambetm/simple_dqn/blob/master/src/plot.py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import constants as C

csv_path = C.stats_csv_path
png_save_path = C.plot_png_path 
figure_height = C.plot_figure_height
figure_width = C.plot_figure_width 
fields = C.plot_stats

# field definitions for numpy
dtype = [
  ("epoch", "int"), 
  ("steps", "int"),
  ("nr_games", "int"),
  ("average_reward", "float"),
  ("total_train_steps", "int"),
  ("meanq", "float"),
  ("meancost", "float"),
  ("cost_per_epoch", "float"),
  ("total_time", "float"),
  ("epoch_time", "float"),
  ("steps_per_second", "float")
]
data = np.loadtxt(csv_path, skiprows = 1, delimiter = ",", dtype = dtype)

# definitions for plot titles
labels = {
  "epoch": "Epoch", 
  "steps": "Number of steps",
  "nr_games": "Number of games",
  "average_reward": "Average reward",
  "total_train_steps": "Exploration steps",
  "meanq": "Average Q-value",
  "meancost": "Average log loss",
  "cost_per_epoch": "Log loss per epoch",
  "total_time": "Total time elapsed",
  "epoch_time": "Phase time",
  "steps_per_second": "Number of steps per second"
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
  plt.legend(["Train"], loc = "best")
  plt.ylabel(labels[field])
  plt.xlabel(labels['epoch'])
  plt.title(labels[field])

plt.tight_layout()

# if png_file argument given, save to file
if png_save_path != "":
  plt.savefig(png_save_path)
else:
  plt.show()