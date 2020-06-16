from game import Game
from joblib import Parallel, delayed
from memory import Memory
from config import MCTS_SIMS
from timeit import default_timer as time
from ISMCTS import ISMCTS as mc

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
        
        for s in states:
            memories.append((s, value))
    
    return memories

def fill_mem(memories):
    # get remaining memory count
    remaining = memories[0].MEMORY_SIZE - len(memories[0].ltmemory)
    
    n_jobs = 4
    
    batch_size = remaining / n_jobs if remaining % n_jobs == 0 else int(remaining / n_jobs) + 1
    
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    
    start = time()
    chunks = executor(delayed(worker)(batch_size, testing_agent(MCTS_SIMS, 'best_player')) for _ in range(n_jobs))
    
    for chunk in chunks:
        for state, value in chunk:
            memories[0].commit_stmemory(state)
            memories[0].stmemory[-1]["value"] = value
    
    memories[0].commit_ltmemory()
    total = time() - start
    print("Total time: {}, time per memory: {}".format(total, total/len(memories[0].ltmemory)))
    
    