from agent import *
from funcs import *
from config import *
import config
from game import *
import loggers as lg
import random
from model import Residual_CNN
import initialise
import pickle
from settings import run_folder, run_archive_folder
from memory import Memory
import sys

arg = sys.argv[1]

game = Game()

if arg == "output_eval":
    memories = []


    if initialise.INITIAL_MEMORY_VERSION == [None] * DECISION_TYPES:
        for i in range(DECISION_TYPES):
            memories.append(Memory(MEMORY_SIZE[i]))
    else:
        for d_t, MEM_VERSION in enumerate(initialise.INITIAL_MEMORY_VERSION):
            print('LOADING MEMORY VERSION ' + str(MEM_VERSION) + '...')
            memories.append(pickle.load(open(
                run_archive_folder + game.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/decision_" + str(d_t) + "_memory" + str(MEM_VERSION).zfill(4) + ".p", "rb")))

            if memories[-1].MEMORY_SIZE < MEMORY_SIZE[d_t]:
                memories[-1].extension(MEMORY_SIZE[d_t])
                
    nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, game.grid_shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS, 0)
    m_tmp = nn.read(game.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION[0])
    nn.model.set_weights(m_tmp.get_weights())
    trained_agent = Agent('trained_agent', game.action_size, config.MCTS_SIMS, config.CPUCT, [nn])
    
    trained_agent.evaluate_accuracy(memories[0].ltmemory, 0)
    
    quit()
                
    

if arg == "pred_test":
    nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + game.grid_shape, [1],\
                          config.HIDDEN_CNN_LAYERS, 0)
    player = Agent('player', game.state_size, config.MCTS_SIMS, config.CPUCT, [nn])
    
    print(player.predict_value(game.gameState))
    
    exit()

if arg == "rollout_test":
    rollout_agent = testing_agent(MCTS_SIMS, 'trained_agent')
    random_agent = User('random_agent')

    if PLAYER_COUNT == 2:
        version_tournament([rollout_agent, random_agent], 1000, lg.logger_tourney)
    else:
        version_tournament([rollout_agent, random_agent, random_agent, random_agent], 400, lg.logger_tourney)

if arg == "nn_test":
    nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, game.grid_shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS, 0)
    m_tmp = nn.read(game.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION[0])
    nn.model.set_weights(m_tmp.get_weights())
    #trained_agent = Agent('trained_agent', game.action_size, config.MCTS_SIMS, config.CPUCT, [nn])
    trained_agent = testing_agent(MCTS_SIMS, 'trained_agent')
    
    random_agent = User('random_agent')
    
    version_tournament([trained_agent, random_agent, random_agent, random_agent], 100, lg.logger_tourney)
    
    quit()
    
if arg == "base_test":
    base = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, game.grid_shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS, 0)
    nn = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, game.grid_shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS, 0)
    
    print("Loading model {}".format(initialise.INITIAL_MODEL_VERSION[0]))
    
    m_tmp = nn.read(game.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION[0])
    nn.model.set_weights(m_tmp.get_weights())
    trained_agent = Agent('trained_agent', game.action_size, config.MCTS_SIMS, config.CPUCT, [nn])
    #trained_agent = testing_agent(MCTS_SIMS, 'trained_agent')
    
    rollout_agent = testing_agent(MCTS_SIMS, 'rollout_agent')
    
    #trained_agent.evaluate()
    
    untrained_agent = Agent('untrained_agent', game.action_size, config.MCTS_SIMS, config.CPUCT, [base])
    #untrained_agent = testing_agent(MCTS_SIMS, 'untrained_agent')
    #version_tournament([trained_agent, untrained_agent, trained_agent, untrained_agent], 400, lg.logger_tourney)
    
    version_tournament([rollout_agent, trained_agent, rollout_agent, trained_agent], 800, lg.logger_tourney)
    
    quit()
    
    

randomization_test = False

if arg == "randomization_test":
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
