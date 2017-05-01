#AI
ai_qtable_learning_rate = .85
ai_qtable_future_discount = .99
ai_qtable_num_episode_length = 100

ai_num_episodes = 1000
ai_batch_size = 32
ai_init_epsilon = 1
ai_final_epsilon = .1
ai_replay_mem_start_size = 50000
ai_replay_mem_total_size = 1000000

#Network
net_restore_path = ""
net_save_path = "SavedSessions/test2"
net_should_save = True

net_learning_rate = 0.00025 #Google DeepMind used this rate 
net_rmsprop_momentum = 0.95
net_rmsprop_epsilon = 0.01
net_height = 84
net_width = 84
net_num_screens = 4
net_n_actions = 6
net_discount_factor = .97
net_runs_till_save = 1000000


#Stats 
stats_csv_path = "Stats/test2.csv"

#Plots
plot_png_path = "Stats/test2.png"
plot_figure_height = 10
plot_figure_width = 20
plot_stats = ["average_reward","nr_games","meancost","cost_per_epoch"]

API_KEY = {
"natasha": 'sk_9Ft7yJrgT2M4k7y7Fe8A',
"hong": 'sk_9Ft7yJrgT2M4k7y7Fe8A', 
}







