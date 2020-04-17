import numpy as np
import logging
import config
from config import PLAYER_COUNT, TEAM_SIZE


from utils import setup_logger
import loggers as lg



import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import tempfile

from copy import deepcopy
class Node():

	def __init__(self, state, id):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = id
		self.render_id = 0
		self.edges = []
		self.inEdges = []	
		self.edge_actions = []	

	def isLeaf(self):
		if len(self.edges) > 0 and not self.state.isEndGame:
			return False
		else:
			return True

	def getVisits(self):
		total = 0
		for edge in self.inEdges:
			total += edge.bandit_stats['V']
		return total

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

		self.bandit_stats = {'V': 0, 'R': 0, 'P':0} 	# Visits, Rewards, parent_visits

				

class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.addNode(root)
		self.converge_count = 0
	
	def __len__(self):
		return len(self.tree)

	# returns list of legal untried actions
	def getUntriedActions(self, node, legal_actions):
		tried_actions = [action for (action, edge) in node.edges if edge.outNode != None]	# list of already taken actions from this node
		existing_untried = [action for (action, edge) in node.edges if edge.outNode == None]

		return [action for action in legal_actions if action not in tried_actions], existing_untried, legal_actions	# list of actions from legal_actions not in tried_actions

	# choose the edge w/ the highest Q+U value until an unexplored edge or terminal state is encountered
	# then return that node
	def moveToLeaf(self, agent):

		breadcrumbs = []
		currentNode = self.root

		if currentNode.edges == []:
			for action in currentNode.state.allowedActions:
				new_edge = Edge(currentNode, None, action)
				currentNode.edges.append((action, new_edge))
	

		done = 0
		value = 0

		while not currentNode.isLeaf():
			#lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			# update the parent visits for each edge from currentNode
			for (action, edge) in currentNode.edges:
				edge.bandit_stats['P'] += 1

			max_ucb = -99999

			untried_actions, existing_untried, legal_actions = self.getUntriedActions(currentNode, currentNode.state.allowedActions)	
			
			if untried_actions == []:	# if all actions have been explored update the stats of currentNode's edges and choose the highest Q+U
				for idx, (action, edge) in enumerate(currentNode.edges):
					if action in legal_actions:	#TODO: make legal actions a dict
						visits = edge.bandit_stats['V']

						#calculate UCB1 value for each edge
						# (total reward / # of visits) + exploration_constant * sqrt(log(# of parent visits)/# of visits)

						ucb_temp = (edge.bandit_stats['R'] / visits) + 0.7 * np.math.sqrt(np.math.log(edge.bandit_stats['P']) / visits)
						#only hold the MAX UCB value and then move on to next (action, edge)
						if ucb_temp > max_ucb:
							max_ucb = ucb_temp
							simulationAction = action
							simulationEdge = edge
			elif not currentNode.isLeaf():	# if there are untried actions choose a random one then get predictions for ALL available actions and update the old predictions
				simulationAction = np.random.choice(untried_actions)
				chosen_edge = None

				for i, action in enumerate(currentNode.state.allowedActions):	# insert new action edge pairs into currentNode.edges
					if action in untried_actions and action not in existing_untried:	# dirty but works
						new_edge = Edge(currentNode, None, action)
						currentNode.edges.append((action, new_edge))

						if action == simulationAction:
							chosen_edge = new_edge

				if chosen_edge == None:
					for (action, edge) in currentNode.edges:
						if action == simulationAction:
							chosen_edge = edge
							break

				# TODO: averaging prior probability with new probability

				#lg.logger_mcts.info('Untried action: %d', simulationAction)


			

			newState, value_tuple, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn

			# if a new action is being explored add the resulting node to the tree
			# and store that node in the corresponding edge's outNode as well as the
			# edge in the node's inEdge
			if untried_actions != []:
				id = newState.id
				
				if id not in self.tree:
					node = Node(newState, id)
					self.addNode(node)
					#lg.logger_mcts.info('added node...%s', node.id)
				else:
					node = self.tree[id]
					#lg.logger_mcts.info('existing node...%s',node.id)
					node.state = newState

				chosen_edge.outNode = node
				node.inEdges.append(chosen_edge)
				simulationEdge = chosen_edge
			if TEAM_SIZE > 1:
				value = value_tuple[newState.playerTurn % TEAM_SIZE]
			else:
				value = value_tuple[newState.playerTurn]

			currentNode = simulationEdge.outNode
			currentNode.state = newState
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		return currentNode, done, breadcrumbs

	def moveToLeaf_rollout(self, agent):
		breadcrumbs = []
		currentNode = self.root
		root_player = currentNode.state.playerTurn
	

		done = 0
		value = 0

		while not currentNode.isLeaf():
			#lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			# update the parent visits for each edge from currentNode
			for (action, edge) in currentNode.edges:
				edge.bandit_stats['P'] += 1

			max_ucb = -99999

			untried_actions, existing_untried, legal_actions = self.getUntriedActions(currentNode, currentNode.state.allowedActions)	
			
			if untried_actions == []:	# if all actions have been explored update the stats of currentNode's edges and choose the highest Q+U
				for idx, (action, edge) in enumerate(currentNode.edges):
					if action in legal_actions:	#TODO: make legal actions a dict
						visits = float(edge.bandit_stats['V'])

						#calculate UCB1 value for each edge
						# (total reward / # of visits) + exploration_constant * sqrt(log(# of parent visits)/# of visits)

						ucb_temp = (float(edge.bandit_stats['R']) / visits) + 0.7 * np.math.sqrt(np.math.log(edge.bandit_stats['P']) / visits)
						#only hold the MAX UCB value and then move on to next (action, edge)
						if ucb_temp > max_ucb:
							max_ucb = ucb_temp
							simulationAction = action
							simulationEdge = edge
			elif not currentNode.isLeaf():	# if there are untried actions choose a random one then get predictions for ALL available actions and update the old predictions
				simulationAction = np.random.choice(untried_actions)
				chosen_edge = None

				new_untried = frozenset(untried_actions) - frozenset(existing_untried)

				for action in new_untried:
					new_edge = Edge(currentNode, None, action)
					currentNode.edges.append((action, new_edge))

					if action == simulationAction:
						chosen_edge = new_edge

				if chosen_edge == None:
					for (action, edge) in currentNode.edges:
						if action == simulationAction:
							chosen_edge = edge
							break

				# TODO: averaging prior probability with new probability

				#lg.logger_mcts.info('Untried action: %d', simulationAction)


			
			try:
				newState, value_tuple, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
				while len(newState.allowedActions) < 2 and not newState.isEndGame:
					if len(newState.allowedActions) == 1:
						simulationAction = newState.allowedActions[0]
					else:
						simulationAction = -1
					newState, value_tuple, done = newState.takeAction(simulationAction)
			except:
				print("error")
			# if a new action is being explored add the resulting node to the tree
			# and store that node in the corresponding edge's outNode as well as the
			# edge in the node's inEdge
			if untried_actions != []:
				if newState.playerTurn == root_player:
					state_id = newState.id
				else:
					state_id = gen_id(newState, root_player) #TODO: here
				
				if state_id not in self.tree:
					node = Node(newState, state_id)
					self.addNode(node)
					#lg.logger_mcts.info('added node...%s', node.id)
				else:
					self.converge_count += 1
					node = self.tree[state_id]
					#lg.logger_mcts.info('existing node...%s',node.id)
					node.state = newState

				chosen_edge.outNode = node
				node.inEdges.append(chosen_edge)
				simulationEdge = chosen_edge
			if TEAM_SIZE > 1:
				value = value_tuple[newState.playerTurn % TEAM_SIZE]
			else:
				value = value_tuple[newState.playerTurn]

			currentNode = simulationEdge.outNode
			currentNode.state = newState
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		#Rollout
		state = currentNode.state
		value_tuple = state.value
		terminal = state.isEndGame
		while not terminal:
			if state.allowedActions == []:
				temp_action = -1
			else:
				temp_action = np.random.choice(state.allowedActions)
			state, value_tuple, terminal = state.takeAction(temp_action)

		if TEAM_SIZE > 1:
			value = value_tuple[currentNode.state.playerTurn % TEAM_SIZE]
		else:
			value = value_tuple[currentNode.state.playerTurn]

		if not currentNode.state.isEndGame:
			existing_actions = []
			for (action, _) in currentNode.edges:
				existing_actions.append(action)

			for action in currentNode.state.allowedActions:
				if action not in existing_actions:
					currentNode.edges.append((action, Edge(currentNode, 0, action)))

		return currentNode, value, done, breadcrumbs

	
	
	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer or (TEAM_SIZE > 1 and playerTurn % TEAM_SIZE == currentPlayer % TEAM_SIZE):	# added the or to set the values to positive for currentPlayer's partner
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

	def backFill_bandit(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		if TEAM_SIZE > 1:
			currentPlayer = currentPlayer % TEAM_SIZE

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if TEAM_SIZE > 1:
				if playerTurn % TEAM_SIZE == currentPlayer:	# added the or to set the values to positive for currentPlayer's partner
					direction = 1
				else:
					direction = -1
			else:
				if playerTurn == currentPlayer:	# added the or to set the values to positive for currentPlayer's partner
					direction = 1
				else:
					direction = -1

			edge.bandit_stats['V'] += 1
			edge.bandit_stats['R'] += direction * value

			edge.outNode.state.render(lg.logger_mcts)

	def render(self, sims='ERROR'):
		G = nx.DiGraph()
		edges = self.BFS()
		for edge in edges:
			G.add_edge(*edge[0], label=edge[1])
			"""G.nodes[edge[0][0]]['label'] = edge[0][0]
			G.nodes[edge[0][1]]['label'] = edge[0][1]"""
		#G.add_edges_from(edges)
		#print(G.nodes(data=True))
		p=nx.drawing.nx_pydot.to_pydot(G)
		#p.view(tempfile.mktemp('.gv'))  
		p.write_png('ISMCTS_' + str(sims) + '.png')
		
		#p.write('./'

	def addNode(self, node):
		node.render_id = len(self.tree)
		self.tree[node.id] = node

# ID's will all be based on the public information from the perspective of the root player
def gen_id(state, root_player):
	return state.get_public_info(root_player)