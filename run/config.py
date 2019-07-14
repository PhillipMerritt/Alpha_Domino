#### GAME PARAMETERS
PLAYER_COUNT = 4
TEAM_SIZE = 2
DECISION_TYPES = 3

#### SELF PLAY
EPISODES = 10
MCTS_SIMS = 20
RANDOMIZED_SIM_LOOPS = 2
MEMORY_SIZE = 500	# default was 30000 which would take 1500ish episodes to reach
TURNS_UNTIL_TAU0 = 15 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
	{'filters':8, 'kernel_size': (4,4)}
	 , {'filters':8, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_EPISODES = 15
SCORING_THRESHOLD = 1.3