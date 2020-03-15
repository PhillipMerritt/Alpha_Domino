from agent import *
from funcs import *
from config import *
from game import *
import loggers as lg

def print_problem(state):
    temp = []

    for i in range(PLAYER_COUNT):
        if i == state.playerTurn:
            continue
        temp.extend(state.hands[i])
    
    temp.extend(state.queue)

    temp = sorted(temp)
    hidden_doms = set([INDEX2TUP[dom] for dom in temp])

    print('Active Player: {0}'.format(state.playerTurn))

    for i in range(PLAYER_COUNT):
        if i == state.playerTurn:
            continue
        print('Player {0}\n\tHand Size: {1}\n\tClues: {2}'.format(i, len(state.hands[i]), str(state.clues[i])))

    print("Hidden Dominos: {0}".format(hidden_doms))


rollout_test = True
game = Game()

if rollout_test:
    rollout_agent = testing_agent(MCTS_SIMS, 'rollout_agent', game.action_size)
    random_agent = User('random_agent', game.state_size, game.action_size)

    version_tournament([rollout_agent, random_agent], 1000, lg.logger_tourney)

randomization_test = False

if randomization_test: 
    while 1:
        game.reset()
        while not game.gameState.isEndGame:

            if len(game.gameState.queue) > 0:
                print_problem(game.gameState)
            game.step(np.random.choice(game.gameState.allowedActions))
