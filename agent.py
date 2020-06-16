# %matplotlib inline
from timeit import default_timer as timer#timer 
from collections import defaultdict


import numpy as np
import random

from ISMCTS import ISMCTS as mc
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
    def __init__(self, name):
        self.name = name

    def act(self, state, tau):
        state.user_print()
        action = int(input('Enter your chosen action: '))
        
        return action


class Agent():
    def __init__(self, name, action_size, mcts_simulations, cpuct, model):
        self.name = name

        self.action_size = action_size
      
        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model
        self.decision_types = len(model)

        self.train_overall_loss = []
        self.val_overall_loss = []
        
        for i in range(config.DECISION_TYPES):
            self.train_overall_loss.append([])
            self.val_overall_loss.append([])


    def act(self, state, epsilon):
        child_nodes = mc(state, self.MCTSsimulations, agent=self)
        
        if np.random.random() > epsilon:    # if random number is greater than epsilon perform ISMCTS
            return self.chooseAction(child_nodes, True)
        else:   # else choose randomly
            return self.chooseAction(child_nodes, False)

    # Gets a value prediction for the given state and returns it
    def predict_value(self, state):
        decision_type = state.decision_type
        # predict the leaf
        inputToModel = np.array([self.model[decision_type].convertToModelInput(state)])

        preds = self.model[decision_type].predict(inputToModel)

        value = preds[0]

        return value

    def chooseAction(self, child_nodes, deterministic):
        if deterministic:
            max_vists = max(child_nodes, key = lambda c: c.visits).visits
            actions = [c.move for c in child_nodes if c.visits == max_vists]
            action = random.choice(actions)
        else:
            pi = np.zeros(self.action_size[0], dtype=np.integer)
            for child in child_nodes:
                pi[child.move] = child.visits
            
            pi = pi / (np.sum(pi) * 1.0)
            
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0]            
            
        return action
    
    def evaluate(self, ltmemory, d_t):
        minibatch = random.sample(ltmemory, 10)
        #minibatch = ltmemory
        
        predictions = [self.predict_value(row['state']) for row in minibatch]
        targets = [row['value'] for row in minibatch]
        
        for i, pred in enumerate(predictions):
            winner = np.argmax(pred)
            print("Pred: {}, Winner: {}, Same: {}".format(pred, winner, winner == np.argmax(targets[i])))
            
    def evaluate_accuracy(self, ltmemory, d_t):
        #minibatch = random.sample(ltmemory, 9000)
        minibatch = ltmemory
        
        predictions = [self.predict_value(row['state']) for row in minibatch]
        targets = [row['value'] for row in minibatch]
        
        count = 0
        for i, pred in enumerate(predictions):
            if np.argmax(pred) == np.argmax(targets[i]):
                count += 1
        
        print("Accuracy: {}".format(count / len(minibatch)))      

    def replay(self, ltmemory,d_t):
        lg.logger_mcts.info('******RETRAINING MODEL******')
        
            
        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(int(len(ltmemory) * .01), len(ltmemory)))
            
            training_states = np.array([self.model[d_t].convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])}

            fit = self.model[d_t].fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1,\
                                    validation_split=0, batch_size=32)
            lg.logger_mcts.info('D_T {0}: NEW LOSS {1}'.format(d_t, fit.history))
            
            if i == 0:
                init_loss = fit.history['loss'][0]
            
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
        print("Loss reduction: {}".format(init_loss - fit.history['loss'][0]))

class testing_agent(Agent):
    def __init__(self, mcts_simulations, name):
        self.name = name
        self.MCTSsimulations = mcts_simulations
        self.cpuct = 0
    
    def act(self, state, epsilon):
        child_nodes = mc(state, self.MCTSsimulations)
        action = max(child_nodes, key = lambda c: c.visits).move

        return action
