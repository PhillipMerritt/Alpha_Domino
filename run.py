# -*- coding: utf-8 -*-
# %matplotlib inline
#%load_ext autoreload
#%autoreload 2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
"""config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))"""

import settings
import os

play_vs_self = False    # set this to true to take control of all 4 players
play_vs_agent = True   # set this to true to play against a trained

############ Set debugging to true to delete the log folders every time you run the program
debugging = False

if debugging:
    exists = os.path.isfile(settings.run_folder + 'logs/logger_main.log')
    if exists:
        os.remove(settings.run_folder + 'logs/logger_main.log')

    exists = os.path.isfile(settings.run_folder + 'logs/logger_mcts.log')
    if exists:
        os.remove(settings.run_folder + 'logs/logger_mcts.log')

import numpy as np
np.set_printoptions(suppress=True)
#seed = 808 # np.random.random_integers(0,5000)
#print(seed)
#np.random.seed(seed=seed)

from shutil import copyfile
import random
#py_seed = 967 #random.randint(0,1000)
#print("Python seed: {0}".format(py_seed))
#random.seed(py_seed)
from importlib import reload
import sys


from keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from agent import User
from memory import Memory
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

version_testing = False

if version_testing:
    num_matches = 5
    run_version_1 = 1
    run_version_2 = 2
    mem_version_1 = 1
    mem_version_2 = 1

    num_tournies = 20
    total_games = num_tournies * 30

    player_1_total = 0
    sp_total = 0
    for i in range(num_tournies):
        print("Tourney {0} of {1}".format(i+1,num_tournies))
        scores, memory, points, sp_scores = playMatchesBetweenVersions(env,run_version_1,run_version_2,mem_version_1,mem_version_2,30,lg.logger_tourney,10)
        player_1_total += scores['player1']
        sp_total += sp_scores['sp']
        print('\n')
        win_perc = round(float(player_1_total / ((i+1) * 30)) * 100, 4)
        sp_win_perc = round(float(sp_total / ((i+1) * 30)) * 100, 4)

        print("\n")
        print("Trained agent has a win percentage of {0}% after {1} games.".format(win_perc, ((i+1) * 30)))
        print("Starting player has a win percentage of {0}% after {1} games.".format(sp_win_perc, ((i+1) * 30)))
        print('\n')
    exit(-1)



# If loading an existing neural network, copy the config file to root
if initialise.INITIAL_RUN_NUMBER != None:
    copyfile(run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
             './config.py')

import config
from config import PLAYER_COUNT, DECISION_TYPES, MEMORY_SIZE

######## LOAD MEMORIES IF NECESSARY ########

# the replay function retrains based on memory and there is no
# state to get the decision type from so I'm going to make separate
# memories for now
memories = []


if initialise.INITIAL_MEMORY_VERSION == [None] * DECISION_TYPES:
    for i in range(DECISION_TYPES):
        memories.append(Memory(MEMORY_SIZE))
else:
    for d_t, MEM_VERSION in enumerate(initialise.INITIAL_MEMORY_VERSION):
        print('LOADING MEMORY VERSION ' + str(MEM_VERSION) + '...')
        memories.append(pickle.load(open(
            run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/decision_" + str(d_t) + "_memory" + str(MEM_VERSION).zfill(4) + ".p", "rb")))

######## LOAD MODEL IF NECESSARY ########

current_NN = []
best_NN = []

# create an untrained neural network objects from the config file
for i in range(DECISION_TYPES):
    current_NN.append(Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + env.grid_shape, env.action_size[i],
                          config.HIDDEN_CNN_LAYERS, i))
    best_NN.append(Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + env.grid_shape, env.action_size[i],
                                   config.HIDDEN_CNN_LAYERS, i))


best_player_version = []
# If loading an existing neural netwrok, set the weights from that model
if initialise.INITIAL_MODEL_VERSION != [None] * DECISION_TYPES:
    for i, version in enumerate(initialise.INITIAL_MODEL_VERSION):
<<<<<<< HEAD
        best_player_version.append(initialise.INITIAL_MODEL_VERSION[i])
        print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION[i]) + '...')
=======
        best_player_version.append(version)
        print('LOADING MODEL VERSION ' + str(version) + '...')
>>>>>>> a30720733830cfc2add964dd06dc8c433af1bc0e
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

current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

if initialise.INITIAL_ITERATION != None:
    iteration = initialise.INITIAL_ITERATION
else:
    iteration = 0


if play_vs_self:
    reload(lg)
    reload(config)
    user_players = []
    user_players.append(User('User1', env.state_size, env.action_size))
    user_players.append(User('User2', env.state_size, env.action_size))
    user_players.append(User('User3', env.state_size, env.action_size))
    user_players.append(User('User4', env.state_size, env.action_size))

    playMatches(user_players,1,lg.logger_main,500)

if play_vs_agent:
    players = []
    players.append(User('User1', env.state_size, env.action_size))
    players.append(best_player)
    players.append(User('User2', env.state_size, env.action_size))
    players.append(best_player)

    playMatches(players,1,lg.logger_main,0)
    exit(0)

trained = False

while 1:

    iteration += 1
    reload(lg)
    reload(config)

    print('ITERATION NUMBER ' + str(iteration))

    lg.logger_main.info('BEST PLAYER VERSION: {0}'.format(best_player_version))
    print('BEST PLAYER VERSION ' + str(best_player_version))

    ######## CREATE LIST OF PLAYERS #######
    # for training it is just 4 copies of best_player
    best_players = []
    for i in range(PLAYER_COUNT):
        best_players.append(best_player)

    ######## SELF PLAY ########
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memories, _ = playMatches(best_players, config.EPISODES, lg.logger_main,
                                  deterministic_play=False, memory=memories)
    print('\n')
    
    full_memory = True

    for d_t,memory in enumerate(memories):
        memory.clear_stmemory()

        if len(memory.ltmemory) >= config.MIN_MEMORY_SIZE:
            trained = True
            ######## RETRAINING ########
            print('RETRAINING...')
            current_player.replay(memory.ltmemory,d_t)
            print('')
            
            if iteration % 5 == 0:
                pickle.dump(memory, open(run_folder + "memory/decision_" + str(d_t) + "_memory" + str(iteration).zfill(4) + ".p", "wb"))

            lg.logger_memory.info('====================')
            lg.logger_memory.info('NEW MEMORIES')
            lg.logger_memory.info('====================')

            memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))

            for s in memory_samp:
                current_value, current_probs, _ = current_player.get_preds(s['state'],d_t)
                best_value, best_probs, _ = best_player.get_preds(s['state'],d_t)

                lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']])
                lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in current_probs])
                lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in best_probs])
                lg.logger_memory.info('ID: %s', s['state'].id)
                lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model[d_t].convertToModelInput(s['state']))

                s['state'].render(lg.logger_memory)
            
            if len(memory.ltmemory) < MEMORY_SIZE:
                full_memory = False
        else:
            full_memory = False

    if full_memory and MEMORY_SIZE < config.MAX_MEMORY_SIZE:
        print("extending memory!")

        MEMORY_SIZE += config.MEM_INCREMENT

        print("new mem size: {0}".format(MEMORY_SIZE))
        for memory in memories:
            memory.extension(MEMORY_SIZE)
    
    if trained:
        ######## TOURNAMENT ########
        print('TOURNAMENT...')
        # this is fairly specific to Texas42
        # players across from each other are on a team
        # in a 2 player game this tournament would be against the best player and the current player
        # so instead I made an list of players where two randomly sampled best_players are across from eachother
        # and 2 copies of the current player are across from each other
        #best_players = np.random.shuffle(best_players)
        if iteration < 5:
            matches = 3
        elif iteration < 10:
            matches = 5
        elif iteration < 20:
            matches = 7
        else:
            matches = 9

        tourney_players = [best_players[0],current_player,best_players[1],current_player]

        scores, _, points = playMatches(tourney_players, matches, lg.logger_tourney,
                                                deterministic_play=True, memory=[None,None,None])
        print('\nSCORES')
        print(scores)
        print('\n\n')

        # if the current player is significantly better than the best_player replace the best player
        if scores['current_player'] > scores['best_player']:
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