from timeit import default_timer as timer#timer 
import time_keeper as tk
from time_keeper import *

import numpy as np
import random

import loggers as lg
import sys

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config
from config import PLAYER_COUNT, TEAM_SIZE

def playMatchesBetweenVersions(env, run_version_1,run_version_2, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first = 0):
    
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,)+ env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version_1, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())   
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
        
        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version_2, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)

    return (scores, memory, points, sp_scores)


def playMatches(agents, EPISODES, logger, deterministic_play, memory = None, goes_first = 0):
    total_time_avg = 0
    env = Game()
    scores = {"drawn": 0}
    for i in range(PLAYER_COUNT):
        scores[agents[i].name] = 0
    #sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    
    turns = 0

    games_to_win = (config.SCORING_THRESHOLD * EPISODES) / (1 + config.SCORING_THRESHOLD)
    games_to_block = EPISODES - games_to_win

    for e in range(EPISODES):

        logger.info('====================')
        logger.info('EPISODE %d OF %d', e+1, EPISODES)
        logger.info('====================')

        sys.stdout.flush()
        print (str(e+1) + ' ', end='')

        state = env.reset()

        if len(state.allowedActions) == 1:  # if things like bidding at the beginning only give one action go ahead and  automate those w/ env.step
                state, _, _, _ = env.step(state.allowedActions[0], logger)
        
        done = 0
        players = {}
        points = {}

        for i,player in enumerate(agents):
            player.mcts = None
            players[i] = {"agent": player, "name": player.name}
            points[i] = []

        env.gameState.render(logger)
        start_game = timer()

        while done == 0:
            turns = turns + 1 # turns until tao tracker

            d_t = state.decision_type
            #### Run the MCTS algo and return an action
            if not deterministic_play:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)

                # store decision type from state

            if memory != None and memory[d_t] != None:
                ####Commit the move to memory
                memory[d_t].commit_stmemory(env.identities, state, pi)

            if agents[0].name != 'User1':
                logger.info('action: %d', action)
                #for r in range(env.grid_shape[0]):
                    #   logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])
                logger.info('MCTS perceived value for %s: %f', action ,np.round(MCTS_value,2))
                logger.info('NN perceived value for %s: %f', action ,np.round(NN_value,2))
                logger.info('====================')

            ### Do the action
            turn = state.playerTurn

            if players[state.playerTurn]['name'] == 'user':
                state, value, done, _ = env.step(action, logger, True)  # this parameter tells the gameState to print out automated turns for the user's convenience
            else:
                state, value, done, _ = env.step(action, logger) # the value is [1,-1] if team/player 0 won or the opposite if team/player 1 won otherwise it's [0,0]
            
            #env.gameState.render(logger) # moved logger to step so that skipped turns (1 or less action) still get logged

            if done == 1:
                if value[0] == 1:   # assuming there are no draws for now
                    winning_team = 0
                elif value[1] == 1:
                    winning_team = 1
                else:
                    print("value error")
                    exit(0)

                if memory != None:
                    #### If the game is finished, assign the values to the history of moves from the game
                    for d_t in range(config.DECISION_TYPES):
                        if memory[d_t] != None:
                            for move in memory[d_t].stmemory:
                                if move['playerTurn'] % int(PLAYER_COUNT/TEAM_SIZE) == winning_team:
                                    move['value'] = 1
                                else:
                                    move['value'] = -1

                    for i in range(config.DECISION_TYPES):
                        if memory[i] != None:
                            memory[i].commit_ltmemory()
             
                if value[0] == 1:
                    logger.info('%s WINS!', players[0]['name'])
                    scores[players[0]['name']] = scores[players[0]['name']] + 1
                elif value[1] == 1:
                    logger.info('%s WINS!', players[1]['name'])
                    scores[players[1]['name']] = scores[players[1]['name']] + 1
                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    #sp_scores['drawn'] = sp_scores['drawn'] + 1

                for i,pts in enumerate(state.marks):
                    #points[players[state.playerTurn]['name']].append(pts)
                    points[i].append(pts)
                    points[(i+2)%PLAYER_COUNT].append(pts)
        end_game = timer()
        tk.total_game_time = end_game - start_game
        total_time_avg += tk.total_game_time
        #tk.print_ratios(tk.total_game_time, tk.move_to_leaf_time, tk.evaluate_leaf_time, tk.get_preds_time, tk.backfill_time, tk.take_action_time, tk.predict_time)
        tk.total_game_time = 0 
        tk.move_to_leaf_time = 0
        tk.evaluate_leaf_time = 0 
        tk.get_preds_time = 0 
        tk.backfill_time = 0 
        tk.take_action_time = 0 
        tk.predict_time = 0

        # if it is a tournament and if either player has won enough games to win break the loop
        if players[1]['name'] == 'current_player' and (scores['best_player'] > games_to_block or scores['current_player'] > games_to_win):
            break

    print("Avg game time: {0}, Avg # of turns: {1}".format(total_time_avg/EPISODES, int(turns/EPISODES)))
    return (scores, memory, points)
