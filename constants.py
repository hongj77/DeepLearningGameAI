# General
STEPS_PER_EPOCH = 50000 # google rate
SESSION_NAME = "hong_save_test"

#AI
ai_qtable_learning_rate = .85
ai_qtable_future_discount = .99
ai_qtable_num_episode_length = 100
ai_num_episodes = 1000
ai_batch_size = 32
ai_init_epsilon = 1
ai_final_epsilon = .1
ai_replay_mem_total_size = 10000
ai_epsilon_anneal_rate = 1.0/1000000
ai_replay_mem_start_size = 50000
# ai_replay_mem_total_size = 1000000 # this results in memory running out around 400k-600k
ai_replay_mem_total_size = 100000 # maybe try 100k instead

#Network
net_should_save = True
net_should_restore = False # if this is on, then we go to test mode
RESTORE_EPOCH = 6 # change this whenever you need to restore a specific epoch
net_restore_path = "SavedSessions/{}-{}.ckpt".format(SESSION_NAME, RESTORE_EPOCH)
net_save_path = "SavedSessions/{}".format(SESSION_NAME) # path to save the session
net_runs_till_save = STEPS_PER_EPOCH # save every epoch

net_learning_rate = 0.00025 #Google DeepMind used this rate 
net_rmsprop_momentum = 0.95
net_rmsprop_epsilon = 0.01
net_height = 84
net_width = 84
net_num_screens = 4
net_n_actions = 6
net_discount_factor = .97
net_dropout = 1 # probability to keep units

#Stats 
stats_csv_path = "Stats/{}.csv".format(SESSION_NAME)

#Plots
plot_png_path = "Stats/{}.png".format(SESSION_NAME)
plot_figure_height = 10
plot_figure_width = 20
plot_stats = ["average_reward","nr_games","meancost","cost_per_epoch", "meanq"]

API_KEY = {
"natasha": 'sk_9Ft7yJrgT2M4k7y7Fe8A',
"hong": 'sk_9Ft7yJrgT2M4k7y7Fe8A', 
}