import numpy as np
from collections import deque

import config
from importlib import reload

class Memory:
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = MEMORY_SIZE
		self.ltmemory = deque(maxlen=MEMORY_SIZE)
		self.stmemory = deque(maxlen=MEMORY_SIZE)

	def commit_stmemory(self, identities, state, actionValues):
		for r in identities(state, actionValues):
			self.stmemory.append({
				'Decision Type': r[0].decision_type
				, 'playerTurn': state.playerTurn
				, 'state': r[0]
				})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def extension(self, new_max):
		temp_deque = deque(maxlen=new_max)
		temp_deque.extend(self.ltmemory)
		self.ltmemory = temp_deque
		self.MEMORY_SIZE = new_max
	

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=self.MEMORY_SIZE)
		