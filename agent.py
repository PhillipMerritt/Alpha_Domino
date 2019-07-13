# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
from config import DECISION_TYPES
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl


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

    def simulate(self):

        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(lg.logger_mcts)
        lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
        leaf.state.render(lg.logger_mcts)

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def act(self, state, tau):
        state = state.CloneAndRandomize()
        d_t = state.decision_type   # store which decision type this will be

        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        randomized_loops = 10
        avg_pi = np.zeros(len(self.mcts.root.edges))
        avg_values = np.zeros(len(self.mcts.root.edges))
        for i in range(randomized_loops):
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** RANDOMIZED HIDDEN INFO %d ******', i + 1)
            lg.logger_mcts.info('***************************')

            #### run the simulation
            for sim in range(self.MCTSsimulations):
                lg.logger_mcts.info('\t***************************')
                lg.logger_mcts.info('\t****** SIMULATION %d ******', sim + 1)
                lg.logger_mcts.info('\t***************************')
                self.simulate()

             #### get action values
            temp_pi, temp_values = self.getAV(1, d_t)

            avg_pi += temp_pi
            avg_values += temp_values
            
            if i < self.randomized_loops:
                    state = state.CloneAndRandomize() # determinize
                    self.mcts.root.state = state


        avg_pi = avg_pi/randomized_loops
        avg_values = avg_values/randomized_loops

        ####pick the action
        action, value = self.chooseAction(avg_pi, avg_values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = -self.get_preds(nextState,d_t)[0]

        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

        return (action, pi, value, NN_value)

    def get_preds(self, state, decision_type):
        decision_type = state.decision_type
        # predict the leaf
        inputToModel = np.array([self.model[decision_type].convertToModelInput(state)])

        preds = self.model[decision_type].predict(inputToModel)
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

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):

        lg.logger_mcts.info('------EVALUATING LEAF------')

        if done == 0:

            value, probs, allowedActions = self.get_preds(leaf.state, leaf.state.decision_type)
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction(action)
                if newState.id not in self.mcts.tree:
                    node = mc.Node(newState)
                    self.mcts.addNode(node)
                    lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
                else:
                    node = self.mcts.tree[newState.id]
                    lg.logger_mcts.info('existing node...%s...', node.id)

                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))

        else:
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

        return ((value, breadcrumbs))

    def getAV(self, tau, decision_type):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size[decision_type], dtype=np.integer)
        values = np.zeros(self.action_size[decision_type], dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)    # divide every value in pi by the sum of all values in pi
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
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
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
                , 'policy_head': np.array([row['AV'] for row in minibatch])}

            fit = self.model[d_t].fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)
            lg.logger_mcts.info('NEW LOSS %s', fit.history)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')

        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(1.0)

        print('\n')
        self.model[d_t].printWeightAverages()

    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds

    def buildMCTS(self, state):
        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def changeRootMCTS(self, state):
        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
