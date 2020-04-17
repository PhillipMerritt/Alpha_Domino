# %matplotlib inline
from timeit import default_timer as timer#timer 
from collections import defaultdict


import numpy as np
import random

import ISMCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
from config import DECISION_TYPES, TEAM_SIZE, PLAYER_COUNT
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl

import time



class User():
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        state.user_print()
        action = int(input('Enter your chosen action: '))
        pi = np.zeros(self.action_size)
        #pi[action] = 1
        value = None
        NN_value = None
        return (action, pi, value, NN_value)


class Agent():
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model
        self.decision_types = len(model)

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

        for i in range(config.DECISION_TYPES):
            self.train_overall_loss.append([])
            self.train_value_loss.append([])
            self.train_policy_loss.append([])
            self.val_overall_loss.append([])
            self.val_value_loss.append([])
            self.val_policy_loss.append([])

    def simulate(self):

        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(lg.logger_mcts)
        lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        ##### MOVE THE LEAF NODE       
        leaf, done, breadcrumbs = self.mcts.moveToLeaf(self)

        leaf.state.render(lg.logger_mcts)

        ##### EVALUATE THE LEAF NODE
        value = self.get_value(leaf.state, done)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill_bandit(leaf, value, breadcrumbs)

    def act(self, state, epsilon):
        pi = np.zeros(self.action_size[0], dtype=np.integer)

        if np.random.random() > epsilon:    # choose best
            states = state.CloneAndRandomize(self.MCTSsimulations)
            state = states.pop()
            d_t = state.decision_type   # store which decision type this will be

            """if self.mcts == None or state.id not in self.mcts.tree:
                self.buildMCTS(state)
            else:
                self.changeRootMCTS(state)"""

            self.buildMCTS(state)

            #### run the simulation
            for sim in range(self.MCTSsimulations):
                if len(state.allowedActions) == 0:
                    print("agent asked to choose from nothing after {0} simulations".format(sim))
                    exit(1)
                lg.logger_mcts.info('***************************')
                lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
                lg.logger_mcts.info('***************************')
                self.simulate()
                if sim < self.MCTSsimulations - 1:
                    state = states.pop() # determinize
                    self.mcts.root.state = state
                #self.mcts.render(sim)

            ####pick the action
            max_visits = 0
            best_actions = []
            for (action, edge) in self.mcts.root.edges:
                pi[action] = edge.bandit_stats['V']
                if edge.bandit_stats['V'] > max_visits:
                    max_visits = edge.bandit_stats['V']
                    best_actions = [action]
                elif edge.bandit_stats['V'] == max_visits:
                    best_actions.append(action)

            return (np.random.choice(best_actions), pi)
        else:   # choose randomly

            return (np.random.choice(state.allowedActions), pi)

        """
        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)
        """
        

    def get_preds(self, state, decision_type):
        decision_type = state.decision_type
        # predict the leaf
        inputToModel = np.array([self.model[decision_type].convertToModelInput(state)])

        start = timer()
        preds = self.model[decision_type].predict(inputToModel)
        end = timer()


        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]

        logits = logits_array[0]

        allowedActions = state.allowedActions

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)  ###put this just before the for?

        return ((value, probs, allowedActions))

    def get_value(self, state, done):
        if done:
            if TEAM_SIZE > 1:
                return state.value[state.playerTurn % TEAM_SIZE]
            else:
                return state.value[state.playerTurn]

        decision_type = state.decision_type

        # predict the leaf

        inputToModel = np.array([self.model[decision_type].convertToModelInput(state)])

        preds = self.model[decision_type].predict(inputToModel)

        return preds[0][0]

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):

        lg.logger_mcts.info('------EVALUATING LEAF------')

        if leaf == self.mcts.root and len(self.mcts.root.edges) > 0:
            print("evaluating root more than once")
            leaf.state.user_print()
            self.mcts.render()
            exit(1)

        if done == 0:
            start = timer()
            value, probs, allowedActions = self.get_preds(leaf.state, leaf.state.decision_type)
            end = timer()

            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newEdge = mc.Edge(leaf, probs[idx], action)
                leaf.edges.append((action, newEdge))

        else:
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

        return ((value, breadcrumbs))

    def getAV(self, tau, decision_type):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size[decision_type], dtype=np.integer)
        values = np.zeros(self.action_size[decision_type], dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau) # TODO: 1/tau ????
            #print(pi[action])
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)    # divide every value in pi by the sum of all values in pi
        return pi, values

    def chooseAction(self, pi, values, epsilon):

        """if tau == 0:"""
        if np.random.random() > epsilon:
            actions = np.argwhere(pi == max(pi))    # same as np.transpose(np.nonzero(a))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory,d_t):
        lg.logger_mcts.info('******RETRAINING MODEL******')

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))
            
            training_states = np.array([self.model[d_t].convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])}

            fit = self.model[d_t].fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)
            lg.logger_mcts.info('D_T {0}: NEW LOSS {1}'.format(d_t, fit.history))

            self.train_overall_loss[d_t].append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            #self.train_value_loss[d_t].append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))

        #plt.plot(self.train_overall_loss[d_t], 'k')
        #plt.plot(self.train_value_loss[d_t], 'k:')

        #plt.legend(['train_overall_loss', 'train_value_loss'], loc='lower left')

        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(.25)

        print('\n')
        self.model[d_t].printWeightAverages()

        print("D_T {0}, Max = {1}, Min = {2}, latest = {3}".format(d_t, max(self.train_overall_loss[d_t]), min(self.train_overall_loss[d_t]), self.train_overall_loss[d_t][-1]))

    def predict(self, inputToModel, batch_size=None):
        preds = self.model.predict(inputToModel,batch_size)
        return preds

    def buildMCTS(self, state):
        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state, state.id)

        for action in state.allowedActions:
            self.root.edges.append((action, mc.Edge(self.root, 0, action)))

        self.mcts = mc.MCTS(self.root, self.cpuct)

    def changeRootMCTS(self, state):
        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]

class testing_agent(Agent):
    def __init__(self, mcts_simulations, name, action_size):
        self.name = name
        self.MCTSsimulations = mcts_simulations
        self.action_size = action_size
        self.mcts = None
        self.cpuct = 0
    
    def act(self, state, epsilon):
        action = mc(state, self.MCTSsimulations)
        """state_generator = state.CloneAndRandomize(self.MCTSsimulations)
        d_t = state.decision_type   # store which decision type this will be

        self.buildMCTS(state)

        for (action, edge) in self.mcts.root.edges:
            edge.bandit_stats['P'] += 1

        #unique_hands = defaultdict(bool)

        #### run the simulation
        for sim, randomized_state in enumerate(state_generator):
            self.mcts.root.state = randomized_state
    


            

            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('***************************')

            self.simulate()
        
        #print('{0} unique out of {1}'.format(len(unique_hands.keys()), self.MCTSsimulations))

        print(self.mcts.converge_count)
        pi = np.zeros(self.action_size[0], dtype=np.integer)


        ####pick the action
        max_visits = 0
        best_actions = []
        for (action, edge) in self.mcts.root.edges:
            pi[action] = edge.bandit_stats['V']

            if edge.bandit_stats['V'] > max_visits:
                 max_visits = edge.bandit_stats['V']
                 best_actions = [action]
            elif edge.bandit_stats['V'] == max_visits:
                best_actions.append(action)
                
        
        
        return (np.random.choice(best_actions), pi)"""

        return action, None

    # Move to leaf, rollout, backfill
    def simulate(self):
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf_rollout(self)

        self.mcts.backFill_bandit(leaf, value, breadcrumbs)

