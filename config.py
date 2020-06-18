ALL_VERSION_TOURNAMENT = False

#### GAME PARAMETERS
PLAYER_COUNT = 4
TEAM_SIZE = 2
DECISION_TYPES = 1

#### SELF PLAY
EPISODES = 200
MCTS_SIMS = 50
ROLLOUT_RATIO = 0.4
MEMORY_SIZE = 1000000
MIN_MEMORY_SIZE = MEMORY_SIZE / 2
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8


#### RETRAINING
TRAINING_LOOPS = 20
BATCH_SIZE = int((EPISODES * 55 * 0.39) /  TRAINING_LOOPS)
#BATCH_SIZE = 128
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 10e-3
MOMENTUM = 0.9


HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	 , {'filters':75, 'kernel_size': (3,3)}
	]

#### EVALUATION
EVAL_EPISODES = 150
SCORING_THRESHOLD = (EVAL_EPISODES * .55) / (EVAL_EPISODES * .45)