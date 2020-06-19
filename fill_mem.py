from game import Game
from joblib import Parallel, delayed
from memory import Memory
from config import MCTS_SIMS
from timeit import default_timer as time
from ISMCTS import ISMCTS as mc
import pickle
import random
class testing_agent():
    def __init__(self, mcts_simulations, name):
        self.name = name
        self.MCTSsimulations = mcts_simulations
        self.cpuct = 0
    
    def act(self, state, epsilon):
        child_nodes = mc(state, self.MCTSsimulations)
        action = max(child_nodes, key = lambda c: c.visits).move

        return action

def worker(count, agent):
    memories = []
    
    env = Game()
    while len(memories) < count:
        state = env.reset()
        states = []
        done = False
        
        while not done:
            states.append(state)
            
            action = agent.act(state, 0)
            
            (state, value, done, _) = env.step(action)
        
        for s in random.sample(states, 5):
            memories.append((s, value))
    
    return memories

def fill_mem(memories):
    # get remaining memory count
    remaining = memories.MEMORY_SIZE - len(memories.ltmemory)
    
    print("Filling {} memories".format(remaining))
    
    n_jobs = 4
    
    batch_size = remaining / n_jobs if remaining % n_jobs == 0 else int(remaining / n_jobs) + 1
    
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    
    start = time()
    chunks = executor(delayed(worker)(batch_size, testing_agent(MCTS_SIMS, 'best_player')) for _ in range(n_jobs))
    
    for chunk in chunks:
        for state, value in chunk:
            memories.commit_stmemory(state)
            memories.stmemory[-1]["value"] = value
    
    memories.commit_ltmemory()
    total = time() - start
    print("Total time: {}, time per memory: {}".format(total, total/len(memories.ltmemory)))
    #print(59*total/len(memories.ltmemory))
    pickle.dump(memories, open("bo3texas42_50k.p", "wb"))
    
    