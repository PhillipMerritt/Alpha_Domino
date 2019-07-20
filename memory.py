import numpy as np
from collections import deque

import config

class Memory:
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = config.MEMORY_SIZE
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	def commit_stmemory(self, identities, state, actionValues):
		for r in identities(state, actionValues):
			self.stmemory.append({
				'AV': r[1]
				, 'Decision Type': r[0].decision_type
				, 'playerTurn': state.playerTurn
				, 'state': r[0]
				})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)
		