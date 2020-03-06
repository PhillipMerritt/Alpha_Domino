from agent import *
from funcs import *
from config import *
from game import *
import loggers as lg

rollout_test = True
game = Game()

if rollout_test:
    rollout_agent = testing_agent(MCTS_SIMS, 'rollout_agent', game.action_size)
    random_agent = User('random_agent', game.state_size, game.action_size)

    version_tournament([rollout_agent, random_agent], EPISODES, lg.logger_tourney)
