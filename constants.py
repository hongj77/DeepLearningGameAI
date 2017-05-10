# General
STEPS_PER_EPOCH = 10000 # temp
RUN_TILL_EPOCH = 2000 # how many to run
SESSION_NAME = "hong_gpu2"

# AI QTable
ai_qtable_learning_rate = .85
ai_qtable_future_discount = .99
ai_qtable_num_episode_length = 100

# AI Network
ai_batch_size = 32
ai_init_epsilon = 1 # google 
ai_final_epsilon = .1 # google
ai_epsilon_anneal_rate = 1.0/1000000 # google
ai_replay_mem_start_size = 50000 # temp
# ai_replay_mem_total_size = 1000000 # this results in memory running out around 400k-600k
ai_replay_mem_total_size = 300000 # maybe try 100k instead

# Network Save
net_should_save = True
net_should_restore = False # if this is on, then we go to test mode
RESTORE_EPOCH = 1 # change this whenever you need to restore a specific epoch
net_restore_path = "SavedSessions/{}-{}.ckpt".format(SESSION_NAME, RESTORE_EPOCH)
net_save_path = "SavedSessions/{}".format(SESSION_NAME) # path to save the session
net_runs_till_save = STEPS_PER_EPOCH # save every epoch

# Network Params
net_learning_rate = 0.00025 # Google DeepMind used this rate 
net_height = 84
net_width = 84
net_num_screens = 4
net_n_actions = 6
net_discount_factor = .99 # google
net_dropout = 1 # probability to keep units. Not using this right now
net_train_rate = 4 # how many steps for 1 train

# Stats 
stats_csv_path = "Stats/{}.csv".format(SESSION_NAME)

# Plots
plot_png_path = "Stats/{}.png".format(SESSION_NAME)

# OpenAI GYM
API_KEY = {
"natasha": 'sk_9Ft7yJrgT2M4k7y7Fe8A',
"hong": 'sk_9Ft7yJrgT2M4k7y7Fe8A', 
}