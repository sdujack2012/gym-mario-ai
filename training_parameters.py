

# traning parameters
input_size = (70, 100, 4)
sample_size = 32
epoch = 1
max_episodes = 2000
training_before_update_target = 5000
max_steps = 6000
experiences_before_training = 3000
esplison = 0.7
esplison_decay = 0.0001
min_esplison = 0.3
render = False
# reward gamma
gamma = 0.9

# prioritized memory replay
e = 0.00001
a = 0.7
beta = 0.4
beta_increment_per_sampling = 0.00001
capacity = 80000
max_priority = 1.0

# state
top_cutoff = 32
bottom_cutoff = 210
frame_size = (70, 100)
stack_size = 4