def self_play_worker(conn):
    import os
    import config
    from config import PLAYER_COUNT, TEAM_SIZE, MEMORY_SIZE
    from memory import Memory
    from settings import run_folder, run_archive_folder
    import initialise
    from game import Game, GameState
    from agent import Agent
    from model import Residual_CNN, import_tf
    import_tf(1024 * 3)
    from shutil import copyfile
    from funcs import playMatches
    import loggers as lg
    import logging
    import random
    
    
    env = Game()
    
    ######## LOAD MODEL IF NECESSARY ########
    # create an untrained neural network objects from the config file
    if len(env.grid_shape) == 2:
        shape = (1,) + env.grid_shape
    else:
        shape = env.grid_shape

    if TEAM_SIZE > 1:
        best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, int(PLAYER_COUNT / TEAM_SIZE),
                                    config.HIDDEN_CNN_LAYERS)
        opponent_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, int(PLAYER_COUNT / TEAM_SIZE),
                                    config.HIDDEN_CNN_LAYERS)
    else:
        best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                                    config.HIDDEN_CNN_LAYERS)
        opponent_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                                    config.HIDDEN_CNN_LAYERS)
    
    best_player_version = 0
    best_NN.model.set_weights(opponent_NN.model.get_weights())
    
    best_player = Agent('best_player', config.MCTS_SIMS, config.CPUCT, best_NN)
    opponent_player = Agent('selected_opponent', config.MCTS_SIMS, config.CPUCT, opponent_NN)
    
    if initialise.INITIAL_ITERATION != None:
        iteration = initialise.INITIAL_ITERATION
    else:
        iteration = 0
    
    memories = Memory(150 * config.EPISODES)
    while 1:
        iteration += 1
        
        
        # request best_NN weights
        conn.send(best_player_version)
        # wait indefinitely for best_NN weights
        conn.poll(None)
        data = conn.recv()
        #print('recieved: {}'.format(data))
        
        # if weights different set weights
        if data:
            best_NN.model.set_weights(data[1])
            best_player_version = data[0]
        
        if len(memories.ltmemory) != 0:    # send new memories (skip first loop)
            conn.send(memories.ltmemory)
        
        memories = Memory(150 * config.EPISODES)
        ######## CREATE LIST OF PLAYERS #######
        # for training it is just 2 copies of the best_player vs. 2 copies of another randomly selected model
        filenames = os.listdir('run/models/')
        filenames = [name for name in filenames if '.h5' == name[-3:]]
        
        if filenames:
            opponent = random.choice(filenames)
            m_tmp = opponent_NN.read_specific('run/models/' + opponent)
            opponent_NN.model.set_weights(m_tmp.get_weights())
            
            self_play_players = []
            for i in range(PLAYER_COUNT):
                if i % 2 == 0:
                    self_play_players.append(best_player)
                else:
                    self_play_players.append(opponent_player)
        else:
            self_play_players = []
            for i in range(PLAYER_COUNT):
                self_play_players.append(best_player)
            
        #print("Version {} randomly selected to play against version {}".format(int(opponent[-7:-3]), best_player_version))
        
        ######## SELF PLAY ########
        #epsilon = init_epsilon - iteration * (init_epsilon / 50.0)
        epsilon = 0
        
        #print('Current epsilon: {}'.format(epsilon))
        print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
        _, memories = playMatches(self_play_players, config.EPISODES, lg.logger_main,
                                    epsilon, memory=memories)
        #print('\n')