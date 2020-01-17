ALL_VERSION_TOURNAMENT = False

#### GAME PARAMETERS
PLAYER_COUNT = 2
TEAM_SIZE = 1
DECISION_TYPES = 1

#### SELF PLAY
EPISODES = 40
MCTS_SIMS = 50
MEMORY_SIZE = [10000]	# default was 30000 which would take 1500ish episodes to reach
MIN_MEMORY_SIZE = 10000
MEM_INCREMENT = 10000
MAX_MEMORY_SIZE = 10000
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