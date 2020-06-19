import numpy as np
from collections import deque

import config
from importlib import reload

class Memory:
    def __init__(self, MEMORY_SIZE):
        self.MEMORY_SIZE = MEMORY_SIZE
        self.ltmemory = deque(maxlen=MEMORY_SIZE)
        self.stmemory = deque(maxlen=MEMORY_SIZE)
    
    def commit_stmemory(self, state, prev_player = None):
        self.stmemory.append({
            'Decision Type': state.decision_type
            , 'prev_player': prev_player
            , 'state': state
        })

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def extension(self, new_max):
        temp_deque = deque(maxlen=new_max)
        temp_deque.extend(list(self.ltmemory))
        self.ltmemory = temp_deque
        self.MEMORY_SIZE = new_max
    

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=self.MEMORY_SIZE)
        