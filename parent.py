from multiprocessing import Process, Pipe
from time import sleep

from self_play_worker import self_play_worker
from retraining_worker import retraining_worker
from evaluation_worker import evaluation_worker


if __name__ == '__main__':
    best_player_version = 0
    current_player_version = 0
    
    def weights_recieved(weights, current_player_version, current_weights):
        # update current_NN with the new weights
        current_weights.update({current_player_version: weights})
        
    def weight_request(conn, best, version, weights):
        # send either the best or current weights depending on the value of best
        if best:
            conn.send((version, weights))
        else:
            conn.send(weights)
        
    def replace_best(best_weights, new_weights):
        # replace best_NN with current_NN
        best_weights = new_weights
    
    def memories_recieved(memories, mems):
        # add new memories to lt_memory
        for mem in mems:
            memories.stmemory.append(mem)
            #memories.stmemory[-1]['value'] = value
        
        memories.commit_ltmemory()
    
    def memory_request(conn, n, sample_size):
        # create n samples of size sample_size and send them
        samples = []
        
        for _ in range(n):
            samples.append(random.sample(memories.ltmemory, sample_size))
        
        conn.send(samples)
        
    from memory import Memory
    import random
    import initialise
    from config import PLAYER_COUNT, TEAM_SIZE, MEMORY_SIZE
    from settings import run_folder, run_archive_folder
    import pickle
    import config
    from game import Game, GameState
    from model import Residual_CNN, import_tf
    import_tf(False)
    from shutil import copyfile
    
    env = Game()

    # If loading an existing neural network, copy the config file to root
    if initialise.INITIAL_RUN_NUMBER != None:
        copyfile(run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
                './config.py')
    
    ######## LOAD MEMORIES IF NECESSARY ########
    if initialise.INITIAL_MEMORY_VERSION == None:
        mem_version = 0
        memories = Memory(MEMORY_SIZE)
    else:
        print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
        memories = pickle.load(open(
            run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/memory" + str(initialise.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))

        mem_version = initialise.INITIAL_MEMORY_VERSION

        if memories.MEMORY_SIZE != MEMORY_SIZE:
            memories.extension(MEMORY_SIZE)
    
    ######## LOAD MODEL IF NECESSARY ########
    # create an untrained neural network objects from the config file
    if len(env.grid_shape) == 2:
        shape = (1,) + env.grid_shape
    else:
        shape = env.grid_shape

    if TEAM_SIZE > 1:
        tmp_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, int(PLAYER_COUNT / TEAM_SIZE),
                            config.HIDDEN_CNN_LAYERS)
    else:
        tmp_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS)
    
    # If loading an existing neural netwrok, set the weights from that model
    if initialise.INITIAL_MODEL_VERSION != None:
        best_player_version = initialise.INITIAL_MODEL_VERSION
        print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
        m_tmp = tmp_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION)
        
        current_weights = {0:m_tmp.get_weights()}
        best_weights = m_tmp.get_weights()
    # otherwise just ensure the weights on the two players are the same
    else:
        current_weights = {0:tmp_NN.model.get_weights()}
        best_weights = tmp_NN.model.get_weights()
    
    # copy the config file to the run folder
    copyfile('./config.py', run_folder + 'config.py')
        
    # start up each child process
    print("Starting self play process") 
    sp_parent_conn, sp_child_conn = Pipe()
    sp_p = Process(target=self_play_worker, name='self_play_process', args=(sp_child_conn,))
    sp_p.start()
    
    print("Starting retraining process") 
    rt_parent_conn, rt_child_conn = Pipe()
    rt_p = Process(target=retraining_worker, name='retraining_process', args=(rt_child_conn,))
    rt_p.start()
    
    print("Starting evaluation process") 
    ev_parent_conn, ev_child_conn = Pipe()
    ev_p = Process(target=evaluation_worker, name='evaluation_process', args=(ev_child_conn,))
    ev_p.start()
    
    prev_mem_version = -1
    
    while 1:
        # check each connection for waiting data
        
        # self_play connection
        if sp_parent_conn.poll():   # poll returns true if there is waiting data
            data = sp_parent_conn.recv()
            
            if type(data) == int:                       # requesting best_NN weights newer than the sent version number
                print("Beginning self play loop")
                
                #print("SP data: {}".format(data))
                
                if best_player_version > data:          # send them if there is a more recent version
                    weight_request(sp_parent_conn, True, best_player_version, best_weights)
                else:   
                    sp_parent_conn.send(False)
            else:                                       # otherwise it is a new batch of memories to store
                print("Recieved {} new memories".format(len(data)))
                mem_version += 1
                memories_recieved(memories, data)
                print("Memory size: {}".format(len(memories.ltmemory)))
                
                if mem_version % 5 == 0 and mem_version != 0:
                    pickle.dump(memories, open(run_folder + "memory/memory" + str(mem_version).zfill(4) + ".p", "wb"))
        
        # retraining connection
        if rt_parent_conn.poll():
            data = rt_parent_conn.recv()
            #print("Retraining data: {}".format(data))
            
            if type(data[0]) == int:     # a tuple with the number and size of memory samples requested
                if mem_version != prev_mem_version:    
                    print("Retraining current_NN")
                    prev_mem_version = mem_version
                    memory_request(rt_parent_conn, n=data[0], sample_size=data[1])
                else:
                    rt_parent_conn.send(False)
            else:                       # otherwise it's new weights for current_NN
                print("Latest loss: {}".format(data[1]))
                current_player_version += 1
                weights_recieved(data[0], current_player_version, current_weights)
        
        # evaluation connection
        if ev_parent_conn.poll():
            data = ev_parent_conn.recv()
            #print("Evaluation data: {}".format(data))
            
            if type(data) == int:                           # requesting current_NN weights newer than the sent version number
                if current_player_version > data:                  # send them if there is a more recent version
                    print("Tournament between current_player {} and best_player {}".format(current_player_version, best_player_version))
                    weight_request(ev_parent_conn, False, best_player_version, best_weights)
                else:
                    ev_parent_conn.send(False)
            else:                                           # otherwise it is just a message indicating the current_player version that won
                print("current_player {} won a tournament against best_player {}".format(data[0], best_player_version))
                print(data[1])
                
                best_player_version += 1
                replace_best(best_weights, data[0])                      # a tournament so store the current_NN weights as the new best_NN weights

        sleep(10)