from agent import *
from funcs import *
from config import *
import config
from game import *
import loggers as lg
import random
from model import Residual_CNN
import initialise

pred_test = False
rollout_test = False
nn_test = True
game = Game()

if pred_test:
    nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + game.grid_shape, [1],\
                          config.HIDDEN_CNN_LAYERS, 0)
    player = Agent('player', game.state_size, config.MCTS_SIMS, config.CPUCT, [nn])
    
    print(player.predict_value(game.gameState))
    
    exit()

if rollout_test:
    rollout_agent = testing_agent(MCTS_SIMS, 'trained_agent', game.action_size)
    random_agent = User('random_agent', game.state_size, game.action_size)

    if PLAYER_COUNT == 2:
        version_tournament([rollout_agent, random_agent], 1000, lg.logger_tourney)
    else:
        version_tournament([rollout_agent, random_agent, random_agent, random_agent], 400, lg.logger_tourney)

if nn_test:
    nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + game.grid_shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS, 0)
    m_tmp = nn.read(game.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION[0])
    nn.model.set_weights(m_tmp.get_weights())
    trained_agent = Agent('trained_agent', game.state_size, game.action_size, config.MCTS_SIMS, config.CPUCT, [nn])
    random_agent = User('random_agent', game.state_size)
    
    version_tournament([trained_agent, random_agent, random_agent, random_agent], 400, lg.logger_tourney)
    
    quit()
    
    

randomization_test = False

if randomization_test:
    count = 0
    for i in range(2000):
        
        game.reset()
        state = game.gameState
        while not state.isEndGame:
            count += 1
            
            if len(state.allowedActions) > 1:
                state, _, _ = state.takeAction(random.choice(state.allowedActions))
            elif len(state.allowedActions) == 1:
                state, _, _ = state.takeAction(state.allowedActions[0])
            else:
                state, _, _ = state.takeAction(-1)
        
    print(count / 2000)
