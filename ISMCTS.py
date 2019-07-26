import numpy as np
import logging
import config
from config import PLAYER_COUNT, TEAM_SIZE


from utils import setup_logger
import loggers as lg

import time_keeper as tk
from time_keeper import *
class Node():

	def __init__(self, state, id):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = id
		self.edges = []
		self.inEdges = []		

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, prior, action, outNode=None):
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.playerTurn
		self.action = action

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}

		self.bandit_stats = {'V': 0, 'R': 0} 	# Visits and Rewards

				

class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	# returns list of legal untried actions
	def getUntriedActions(self, node):
		legal_actions = node.state.allowedActions	# list of possible actions from this state

		tried_actions = [action for (action, edge) in node.edges if edge.outNode != None]	# list of already taken actions from this node
		existing_untried = [action for (action, edge) in node.edges if edge.outNode == None]

		return [action for action in legal_actions if action not in tried_actions], existing_untried, legal_actions	# list of actions from legal_actions not in tried_actions

	# choose the edge w/ the highest Q+U value until an unexplored edge or terminal state is encountered
	# then return that node
	def moveToLeaf(self, agent):

		lg.logger_mcts.info('------MOVING TO LEAF------')

		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0


		while not currentNode.isLeaf():
			lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			maxQU = -99999

			if currentNode == self.root:
				epsilon = config.EPSILON
				nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))	# creates a list of random weights based on the value of ALPHA
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			untried_actions, existing_untried, legal_actions = self.getUntriedActions(currentNode)	
			
			if untried_actions == []:	# if all actions have been explored update the stats of currentNode's edges and choose the highest Q+U
				Nb = 0					# Nb is the total number of actions taken from state

				for action, edge in currentNode.edges:	
					Nb = Nb + edge.stats['N']

				for idx, (action, edge) in enumerate(currentNode.edges):
					if action in legal_actions:	#TODO: make legal actions a dict
						#function to generate U value, CPUCT is exploration factor, epsilon and nu[i] = 0 for non root states
						U = self.cpuct * \
							((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
							np.sqrt(Nb) / (1 + edge.stats['N'])
							
						Q = edge.stats['Q']
						
						lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
							, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
							, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))
						#only hold the MAX Q + U value and then move on to next (action, edgegh)
						if Q + U > maxQU:
							maxQU = Q + U
							simulationAction = action
							simulationEdge = edge

				lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)
			elif not currentNode.isLeaf():	# if there are untried actions choose a random one then get predictions for ALL available actions and update the old predictions
				simulationAction = np.random.choice(untried_actions)
				_, probs, _ = agent.get_preds(currentNode.state, currentNode.state.decision_type)

				chosen_edge = None

				for i, action in enumerate(currentNode.state.allowedActions):	# insert new action edge pairs into currentNode.edges
					if action in untried_actions and action not in existing_untried:	# dirty but works
						new_edge = Edge(currentNode, probs[i], action)
						currentNode.edges.append((action, new_edge))

						if action == simulationAction:
							chosen_edge = new_edge

				if chosen_edge == None:
					for (action, edge) in currentNode.edges:
						if action == simulationAction:
							chosen_edge = edge
							break

				# TODO: averaging prior probability with new probability

				lg.logger_mcts.info('Untried action: %d', simulationAction)


			

			start = timer()
			newState, value_tuple, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
			end = timer()
			tk.take_action_time += end - start

			# if a new action is being explored add the resulting node to the tree
			# and store that node in the corresponding edge's outNode as well as the
			# edge in the node's inEdge
			if untried_actions != []:
				if newState.playerTurn == self.root.playerTurn:
					id = newState.id
				else:
					id = gen_id(chosen_edge, self.root.playerTurn)
				
				if id not in self.tree:
					node = Node(newState, id)
					self.addNode(node)
					lg.logger_mcts.info('added node...%s', node.id)
				else:
					node = self.tree[id]
					lg.logger_mcts.info('existing node...%s',node.id)
					node.state = newState

				chosen_edge.outNode = node
				node.inEdges.append(chosen_edge)
				simulationEdge = chosen_edge

			value = value_tuple[newState.playerTurn % TEAM_SIZE]

			currentNode = simulationEdge.outNode
			currentNode.state = newState
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		return currentNode, value, done, breadcrumbs

	def moveToLeaf_rollout(self, agent):
		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0


		while not currentNode.isLeaf():
			lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			max_ucb = -99999

			untried_actions, existing_untried, legal_actions = self.getUntriedActions(currentNode)	
			
			if untried_actions == []:	# if all actions have been explored update the stats of currentNode's edges and choose the highest Q+U
				for idx, (action, edge) in enumerate(currentNode.edges):
					if action in legal_actions:	#TODO: make legal actions a dict
						visits = edge.bandit_stats['V']						
						ucb_temp = (edge.bandit_stats['R'] / visits) \
							+ 0.7 * np.math.sqrt(np.math.log(current_node.inEdges[-1].bandit_stats['V']) / visits)
						#only hold the MAX UCB value and then move on to next (action, edge)
						if ucb_ temp > max_ucb:
							max_ucb = ucb_temp
							simulationAction = action
							simulationEdge = edge

			elif not currentNode.isLeaf():	# if there are untried actions choose a random one then get predictions for ALL available actions and update the old predictions
				simulationAction = np.random.choice(untried_actions)
				chosen_edge = None

				for i, action in enumerate(currentNode.state.allowedActions):	# insert new action edge pairs into currentNode.edges
					if action in untried_actions and action not in existing_untried:	# dirty but works
						new_edge = Edge(currentNode, probs[i], action)
						currentNode.edges.append((action, new_edge))

						if action == simulationAction:
							chosen_edge = new_edge

				if chosen_edge == None:
					for (action, edge) in currentNode.edges:
						if action == simulationAction:
							chosen_edge = edge
							break

				# TODO: averaging prior probability with new probability

				lg.logger_mcts.info('Untried action: %d', simulationAction)


			

			start = timer()
			newState, value_tuple, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
			end = timer()
			tk.take_action_time += end - start

			# if a new action is being explored add the resulting node to the tree
			# and store that node in the corresponding edge's outNode as well as the
			# edge in the node's inEdge
			if untried_actions != []:
				if newState.playerTurn == self.root.playerTurn:
					id = newState.id
				else:
					id = gen_id(chosen_edge, self.root.playerTurn)
				
				if id not in self.tree:
					node = Node(newState, id)
					self.addNode(node)
					lg.logger_mcts.info('added node...%s', node.id)
				else:
					node = self.tree[id]
					lg.logger_mcts.info('existing node...%s',node.id)
					node.state = newState

				chosen_edge.outNode = node
				node.inEdges.append(chosen_edge)
				simulationEdge = chosen_edge

			value = value_tuple[newState.playerTurn % TEAM_SIZE]

			currentNode = simulationEdge.outNode
			currentNode.state = newState
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		#Rollout
		state = currentNode.state
		while not state.isEndGame:
			temp_action = np.random.choice(state.allowedActions)
			state, _, _ = state.takeAction()
		return currentNode, value, done, breadcrumbs

	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer or playerTurn == (currentPlayer + 2) % 4:	# added the or to set the values to positive for currentPlayer's partner
				direction = 1
			else:
				direction = -1

			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value * direction
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
				, value * direction
				, playerTurn
				, edge.stats['N']
				, edge.stats['W']
				, edge.stats['Q']
				)

			edge.outNode.state.render(lg.logger_mcts)

	def addNode(self, node):
		self.tree[node.id] = node

# ID's will all be based on the perspective of the root player
# i.e. the ID will either be the actual state ID if the root player is the
# current player in that state or it will be the ID of the last state where
# the root player was the current player followed by the series of moves that
# led to this state
def gen_id(inEdge, root_player):
	action_trail = [inEdge.action]
	temp_node = inEdge.inNode
	while temp_node.playerTurn != root_player:
		action_trail.append(temp_node.inEdges[0].action)
		temp_node = temp_node.inEdges[0].inNode
	
	id = temp_node.state.id
	while action_trail != []:
		id += '|' + str(action_trail.pop())

	return id