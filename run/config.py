#### GAME PARAMETERS
PLAYER_COUNT = 4
TEAM_SIZE = 2
DECISION_TYPES = 2

#### SELF PLAY
EPISODES = 5
MCTS_SIMS = 50
RANDOMIZED_SIM_LOOPS = 2
MEMORY_SIZE = [4000, 1000]	# default was 30000 which would take 1500ish episodes to reach
MIN_MEMORY_SIZE = 1000
MEM_INCREMENT = 250
MAX_MEMORY_SIZE = 5000
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
	{'filters':32, 'kernel_size': (4,4)}
	 , {'filters':32, 'kernel_size': (4,4)}
	 , {'filters':32, 'kernel_size': (4,4)}
	 , {'filters':32, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_EPISODES = 25
SCORING_THRESHOLD = 1.3