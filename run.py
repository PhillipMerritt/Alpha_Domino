# -*- coding: utf-8 -*-
# %matplotlib inline
#%load_ext autoreload
#%autoreload 2
import tensorflow as tf

import settings
import os
import random
import numpy as np
np.set_printoptions(suppress=True)
#seed = 808 # np.random.random_integers(0,5000)
#print(seed)
#np.random.seed(seed=seed)

#py_seed = 967 #random.randint(0,1000)
#print("Python seed: {0}".format(py_seed))
#random.seed(py_seed)

from shutil import copyfile
from importlib import reload
import sys
from keras.utils import plot_model
from game import Game, GameState
from agent import Agent, testing_agent
from agent import User
from memory import Memory
from model import Residual_CNN
from funcs import *
import loggers as lg
import logging
from settings import run_folder, run_archive_folder
import initialise
import pickle
import config
from config import PLAYER_COUNT, TEAM_SIZE, DECISION_TYPES, MEMORY_SIZE, ALL_VERSION_TOURNAMENT
from fill_mem import fill_mem

play_vs_self = False    # set this to true to take control of all 4 players
play_vs_agent = False   # set this to true to play against a trained
version_testing = False # pit two models version against eachother 
ismcts_agent_test = False   # test against the non-NN implementation of ISMCTS


############ Set debugging to true to delete the log folders every time you run the program
debugging = False

if debugging:
    exists = os.path.isfile(settings.run_folder + 'logs/logger_main.log')
    if exists:
        os.remove(settings.run_folder + 'logs/logger_main.log')

    exists = os.path.isfile(settings.run_folder + 'logs/logger_mcts.log')
    if exists:
        os.remove(settings.run_folder + 'logs/logger_mcts.log')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

# If loading an existing neural network, copy the config file to root
if initialise.INITIAL_RUN_NUMBER != None:
    copyfile(run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
             './config.py')

######## LOAD MEMORIES IF NECESSARY ########

# the replay function retrains based on memory and there is no
# state to get the decision type from so I'm going to make separate
# memories for now
memories = []


if initialise.INITIAL_MEMORY_VERSION == [None] * DECISION_TYPES:
    for i in range(DECISION_TYPES):
        memories.append(Memory(MEMORY_SIZE[i]))
else:
    for d_t, MEM_VERSION in enumerate(initialise.INITIAL_MEMORY_VERSION):
        print('LOADING MEMORY VERSION ' + str(MEM_VERSION) + '...')
        memories.append(pickle.load(open(
            run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/decision_" + str(d_t) + "_memory" + str(MEM_VERSION).zfill(4) + ".p", "rb")))

        if memories[-1].MEMORY_SIZE < MEMORY_SIZE[d_t]:
            memories[-1].extension(MEMORY_SIZE[d_t])
######## LOAD MODEL IF NECESSARY ########

current_NN = []
best_NN = []

# create an untrained neural network objects from the config file
for i in range(DECISION_TYPES):
    if len(env.grid_shape) == 2:
        shape = (1,) + env.grid_shape
    else:
        shape = env.grid_shape
    
    if TEAM_SIZE > 1:
        current_NN.append(Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT / TEAM_SIZE,
                            config.HIDDEN_CNN_LAYERS, i))
        best_NN.append(Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT / TEAM_SIZE,
                                    config.HIDDEN_CNN_LAYERS, i))
    else:
        current_NN.append(Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS, i))
        best_NN.append(Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                                    config.HIDDEN_CNN_LAYERS, i))

best_player_version = []
# If loading an existing neural netwrok, set the weights from that model
if initialise.INITIAL_MODEL_VERSION != [None] * DECISION_TYPES:
    for i, version in enumerate(initialise.INITIAL_MODEL_VERSION):
        best_player_version.append(initialise.INITIAL_MODEL_VERSION[i])
        print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION[i]) + '...')
        m_tmp = best_NN[i].read(env.name, initialise.INITIAL_RUN_NUMBER, version)
        current_NN[i].model.set_weights(m_tmp.get_weights())
        best_NN[i].model.set_weights(m_tmp.get_weights())
# otherwise just ensure the weights on the two players are the same
else:
    for i in range(DECISION_TYPES):
        best_player_version.append(0)
        best_NN[i].model.set_weights(current_NN[i].model.get_weights())

# copy the config file to the run folder
copyfile('./config.py', run_folder + 'config.py')

for i in range(DECISION_TYPES):
    plot_model(current_NN[i].model, to_file=run_folder + 'models/decision_' + str(i) + '_model.png', show_shapes=True)

print('\n')

######## CREATE THE PLAYERS ########

current_player = Agent('current_player', env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

rollout_first = False

if initialise.INITIAL_ITERATION != None:
    iteration = initialise.INITIAL_ITERATION
else:
    iteration = 0
    rollout_first = True
    best_player = testing_agent(config.MCTS_SIMS, 'best_player')

"""if len(memories[0].ltmemory) < MIN_MEMORY_SIZE:
    fill_mem(memories)
    pickle.dump(memories[0], open(run_folder + "memory/decision_" + str(0) + "_memory" + str(iteration).zfill(4) + ".p", "wb"))"""

trained = False
epsilon = init_epsilon = 0.70

while 1:

    iteration += 1
    
    
    reload(lg)
    reload(config)

    print('ITERATION NUMBER ' + str(iteration))

    lg.logger_main.info('BEST PLAYER VERSION: {0}'.format(best_player_version))
    print('BEST PLAYER VERSION ' + str(best_player_version))

    ######## CREATE LIST OF PLAYERS #######
    # for training it is just 4 copies of best_player
    best_players = [best_player for i in range(PLAYER_COUNT)]

    ######## SELF PLAY ########
    epsilon = init_epsilon - iteration * (init_epsilon / 50.0)
    
    print('Current epsilon: {}'.format(epsilon))
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memories = playMatches(best_players, config.EPISODES, lg.logger_main,
                                  epsilon, memory=memories)
    print('\n')
    
    full_memory = True

    for d_t,memory in enumerate(memories):
        memory.clear_stmemory()

        if len(memory.ltmemory) >= MIN_MEMORY_SIZE:
            #set_learning_phase(1) # tell keras backend that the model will be learning now

            trained = True
            ######## RETRAINING ########
            print('RETRAINING...')
            current_player.replay(memory.ltmemory,d_t)
            current_player.evaluate(memory.ltmemory,d_t)
            print('')
            
    if iteration != 0 and iteration % 3 == 0:
                pickle.dump(memory, open(run_folder + "memory/decision_" + str(d_t) + "_memory" + str(iteration).zfill(4) + ".p", "wb"))
    
    if trained:
        ######## TOURNAMENT ########
        print('TOURNAMENT...')
        # this is fairly specific to Texas42
        # players across from each other are on a team
        # in a 2 player game this tournament would be against the best player and the current player
        # so instead I made an list of players where two randomly sampled best_players are across from eachother
        # and 2 copies of the current player are across from each other
        
        tourney_players = []
        if TEAM_SIZE > 1:
            for i in range(int(PLAYER_COUNT / TEAM_SIZE)): # for each team
                for k in range(TEAM_SIZE): # alternate adding best_players and current_players up to the TEAM_SIZE
                    if k % 2 == 0:
                        tourney_players.append(best_player)
                    else:
                        tourney_players.append(current_player)
        else:
            for i in range(PLAYER_COUNT):
                if i % 2 == 0:
                    tourney_players.append(best_player)
                else:
                    tourney_players.append(current_player)
                    
        scores, _ = playMatches(tourney_players, config.EVAL_EPISODES, lg.logger_tourney,
                                                0.0, None, best_player_version[0] == 0)
        print('\nSCORES')
        print(scores)
        print('\n\n')

        # if the current player is significantly better than the best_player replace the best player
        # the replacement is made by just copying the weights of current_player's nn to best_player's nn
        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            if rollout_first:
                best_player = Agent('best_player', env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
                rollout_first = False
            for i in range(DECISION_TYPES):
                best_player_version[i] = best_player_version[i] + 1
                best_NN[i].model.set_weights(current_NN[i].model.get_weights())
                best_NN[i].write(env.name, best_player_version[i])

    
    mem_size = 'MEMORY SIZE: '
    for i, memory in enumerate(memories):
        mem_size += str(i) + ':' + str(len(memory.ltmemory))
        if i < DECISION_TYPES - 1:
            mem_size += ', '
    print(mem_size)