from timeit import default_timer as timer#timer 



import numpy as np
import random

import loggers as lg
import sys

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config
from config import PLAYER_COUNT, TEAM_SIZE, MIN_MEMORY_SIZE

from collections import defaultdict

def playMatches(agents, EPISODES, logger, epsilon, memory = None, random_agent = False):
    total_time_avg = 0
    env = Game()
    scores = {"drawn": 0}
    for i in range(PLAYER_COUNT):
        scores[agents[i].name] = 0
    #sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    
    turns = 0

    epsilon_step = 1/EPISODES

    games_to_win = (config.SCORING_THRESHOLD * EPISODES) / (1 + config.SCORING_THRESHOLD)
    games_to_block = EPISODES - games_to_win

    for e in range(EPISODES):
        last_game = e + 1

        logger.info('====================')
        logger.info('EPISODE %d OF %d', e+1, EPISODES)
        logger.info('====================')

        sys.stdout.flush()
        print (str(e+1) + ' ', end='')

        state = env.reset()
        
        
        if state.isEndGame:
            print("over before started")
            exit(0)
            
        #print("Starting hands: {0}".format(state.hands))

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
            turns = turns + 1

            if len(state.allowedActions) < 2:
                print("funcs loop, no choices")

            #### Run the MCTS algo and return an action
            if random_agent and players[state.playerTurn]['name'] == 'best_player':
                action = random.choice(state.allowedActions)
            else:
                action = players[state.playerTurn]['agent'].act(state, epsilon)
                
            ### Do the action
            turn = state.playerTurn
            if players[state.playerTurn]['name'] == 'user':
                state, value, done, _ = env.step(action, logger, True)  # this parameter tells the gameState to print out automated turns for the user's convenience
            else:
                state, value, done, _ = env.step(action, logger) # the value is [1,-1] if team/player 0 won or the opposite if team/player 1 won otherwise it's [0,0]
            
            # store state and the player who created that state in short term memory
            if memory != None and memory != None:
                memory.commit_stmemory(state, turn)

            if done:
                if memory:
                    #### If the game is finished, assign the values to the history of moves from the game
                    for move in memory.stmemory:
                        """if TEAM_SIZE > 1:
                            if move['prev_player'] % TEAM_SIZE == winning_team:
                                move['value'] = 1
                            else:
                                move['value'] = -1
                        else:
                            if move['prev_player'] == winning_team:
                                move['value'] = 1
                            else:
                                move['value'] = -1"""
                                
                        move['value'] = value

                    memory.commit_ltmemory()
                else:
                    if value[0] == 0:
                        scores['drawn'] = scores['drawn'] + 1
                    else:
                        winning_team = int(np.argmax(value))
                        if TEAM_SIZE > 1:
                            winning_team = winning_team % TEAM_SIZE
                        scores[players[winning_team]['name']] = scores[players[winning_team]['name']] + 1
                    """if value[0] == 1:
                        logger.info('%s WINS!', players[0]['name'])
                        scores[players[0]['name']] = scores[players[0]['name']] + 1
                    elif value[1] == 1:
                        logger.info('%s WINS!', players[1]['name'])
                        scores[players[1]['name']] = scores[players[1]['name']] + 1
                    else:
                        logger.info('DRAW...')
                        scores['drawn'] = scores['drawn'] + 1"""
        end_game = timer()
        total_time_avg += end_game - start_game
        
        # if it is a tournament and if either player has won enough games to win break the loop
        if not memory and (scores['best_player'] + .5 * scores['drawn'] >= games_to_block or scores['current_player'] >= games_to_win):
            break
        
        # reduce size of epsilon every episode
        epsilon -= epsilon_step

    print("Avg game time: {0}, Avg # of turns: {1}".format(total_time_avg/last_game, int(turns/last_game)))
    return scores, memory   

def version_tournament(agents, EPISODES, logger):
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

        state = env.reset()
        
        
        if state.isEndGame:
            print("over before started")
            exit(0)
            
        #print("Starting hands: {0}".format(state.hands))

        if len(state.allowedActions) == 1:  # if things like bidding at the beginning only give one action go ahead and  automate those w/ env.step
                state, _, _, _ = env.step(state.allowedActions[0], logger)

        
        done = 0
        players = {}

        for i,player in enumerate(agents):
            player.mcts = None
            players[i] = {"agent": player, "name": player.name}

        env.gameState.render(logger)
        start_game = timer()

        while done == 0:
            #print("turn: {0}".format(turns))
            turns = turns + 1 # turns until tao tracker

            if len(state.allowedActions) < 2:
                print("funcs loop, no choices")

            d_t = state.decision_type
            turn = state.playerTurn
            #### Run the MCTS algo and return an action
            if players[turn]["name"] == 'random_agent':
                 action = random.choice(state.allowedActions)
            else:
                action = players[state.playerTurn]['agent'].act(state, 0)


            if action not in state.allowedActions:
                print("error in funcs")
            ### Do the action

            #print("from funcs")
            if players[state.playerTurn]['name'] == 'user':
                state, value, done, _ = env.step(action, logger, True)  # this parameter tells the gameState to print out automated turns for the user's convenience
            else:
                state, value, done, _ = env.step(action, logger) # the value is [1,-1] if team/player 0 won or the opposite if team/player 1 won otherwise it's [0,0]
            
            #env.gameState.render(logger) # moved logger to step so that skipped turns (1 or less action) still get logged

            if done == 1:
                winning_team = int(np.argmax(value))

                scores[players[winning_team]['name']] += 1
                print(scores)
                print('{0}%'.format(100 * scores['trained_agent'] / (e + 1)))
                
        game_time = timer() - start_game
        total_time_avg += game_time
        
        # if it is a tournament and if either player has won enough games to win break the loop
        if players[1]['name'] == 'current_player' and (scores['best_player'] > games_to_block or scores['current_player'] > games_to_win):
            break
    
    print("Avg game time: {0}, Avg # of turns: {1}".format(total_time_avg/EPISODES, int(turns/EPISODES)))
    return scores


# old and untested
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