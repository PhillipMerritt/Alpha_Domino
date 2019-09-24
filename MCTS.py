import numpy as np
import logging
import config
from config import PLAYER_COUNT, TEAM_SIZE

from utils import setup_logger
import loggers as lg

import time_keeper as tk
from time_keeper import *

import networkx as nx
import matplotlib.pyplot as plt
class Node():

	def __init__(self, state):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
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
				

class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	def moveToLeaf(self):

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
				nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

				

			Nb = 0
			for action, edge in currentNode.edges:
				Nb = Nb + edge.stats['N']

			for idx, (action, edge) in enumerate(currentNode.edges):

				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']

				lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
					, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
					, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

				if Q + U > maxQU:
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge

			lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

			start = timer()
			newState, value_tuple, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
			end = timer()
			tk.take_action_time += end - start

			value = value_tuple[newState.playerTurn % TEAM_SIZE]

			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		return currentNode, value, done, breadcrumbs



	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		leaf_team = leaf.state.playerTurn % TEAM_SIZE

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn % 2 == leaf_team:	# added the or to set the values to positive for currentPlayer's partner
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

	def render(self):
		G = nx.DiGraph()
		edges = self.BFS()

		G.add_edges_from(edges)

		plt.subplot(121)

		nx.draw(G, with_labels=True, font_weight='bold')
		plt.subplot(122)

		nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')


	# creates a list of edges using a BFS to use for rendering
	def BFS(self): 
		visited = {}
		edges = []
		# Mark all the vertices as not visited 
		for node in self.tree:
			visited[node.id] = False

		# Create a queue for BFS 
		queue = [] 

		# Mark the source node as  
		# visited and enqueue it 
		queue.append(self.root) 
		visited[self.root.id] = True

		while queue: 

			# Dequeue a vertex from  
			# queue and print it 
			s = queue.pop(0)

			# Get all adjacent vertices of the 
			# dequeued vertex s. If a adjacent 
			# has not been visited, then mark it 
			# visited and enqueue it 
			for i in s.edges:
				edges.append((s.id,i.outNode.id)) 
				if visited[i.outNode.id] == False: 
					queue.append(i.outNode) 
					visited[i.outNode.id] = True

		return edges
