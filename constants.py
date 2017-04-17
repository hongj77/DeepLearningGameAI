#AI
ai_qtable_learning_rate = .85
ai_qtable_future_discount = .99
ai_qtable_num_episode_length = 100

ai_num_episodes = 2000
ai_batch_size = 32
ai_epsilon = .3


#Network
net_restore_path = ""
net_save_path = ""
net_should_save = False 

net_learning_rate = .005
net_height = 84
net_width = 84
net_num_screens = 4
net_n_actions = 6
net_discount_factor = .2
net_runs_till_save = 1000000


#Stats 
stats_csv_path = "Stats/test2.csv"

#Plots
plot_png_path = "Stats/test2.png"
plot_figure_height = 10
plot_figure_width = 20
plot_stats = ["average_reward","nr_games","meancost","cost_per_epoch"]






