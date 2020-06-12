ALL_VERSION_TOURNAMENT = False

#### GAME PARAMETERS
PLAYER_COUNT = 4
TEAM_SIZE = 1
DECISION_TYPES = 1

#### SELF PLAY
EPISODES = 20
MCTS_SIMS = 50
MEMORY_SIZE = [10000]
MIN_MEMORY_SIZE = 5000
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
#BATCH_SIZE = 128
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 10e-3
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.4