def evaluation_worker(conn):
    import config
    from config import PLAYER_COUNT, TEAM_SIZE, MEMORY_SIZE
    import initialise
    from model import Residual_CNN, import_tf
    import_tf(1024 * 3)
    from game import Game
    from agent import Agent
    from memory import Memory
    from funcs import playMatches
    import loggers as lg
    import logging
    import time
    
    
    
    # initialise new test memory
    test_memories = Memory(int(MEMORY_SIZE/10))
    
    env = Game()
    
    # initialise new models
    # create an untrained neural network objects from the config file
    if len(env.grid_shape) == 2:
        shape = (1,) + env.grid_shape
    else:
        shape = env.grid_shape

    if TEAM_SIZE > 1:
        current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, int(PLAYER_COUNT / TEAM_SIZE),
                            config.HIDDEN_CNN_LAYERS)
        best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, int(PLAYER_COUNT / TEAM_SIZE),
                                    config.HIDDEN_CNN_LAYERS)
    else:
        current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS)
        best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                                    config.HIDDEN_CNN_LAYERS)

    current_player_version = 0
    best_player_version = 0
    # If loading an existing neural netwrok, set the weights from that model
    if initialise.INITIAL_MODEL_VERSION != None:
        best_player_version = initialise.INITIAL_MODEL_VERSION
        #print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
        m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION)
        current_NN.model.set_weights(m_tmp.get_weights())
        best_NN.model.set_weights(m_tmp.get_weights())
    # otherwise just ensure the weights on the two players are the same
    else:
        best_NN.model.set_weights(current_NN.model.get_weights())
    
    current_player = Agent('current_player', config.MCTS_SIMS, config.CPUCT, current_NN)
    best_player = Agent('best_player', config.MCTS_SIMS, config.CPUCT, best_NN)
    
    time.sleep(20)
    
    while 1:
        # request current_NN weights
        conn.send(current_player_version)
        # wait indefinitely for current_NN weights
        conn.poll(None)
        data = conn.recv()
        
        if data:
        
            # set current_NN weights
            current_NN.model.set_weights(data)
            current_player_version += 1
            
            # play tournament games
            tourney_players = []
            if TEAM_SIZE > 1:
                for i in range(int(PLAYER_COUNT / TEAM_SIZE)): # for each team
                    for k in range(TEAM_SIZE): # alternate adding best_players and current_players up to the TEAM_SIZE
                        if k % 2 == 0:
                            tourney_players.append(best_player)
                        else:
                            tourney_players.append(current_player)
            else:
                for i in range(PLAYER_COUNT):
                    if i % 2 == 0:
                        tourney_players.append(best_player)
                    else:
                        tourney_players.append(current_player)
                        
            scores, test_memories = playMatches(tourney_players, config.EVAL_EPISODES, lg.logger_tourney,
                                                    0.0, test_memories, evaluation=True)
            test_memories.clear_stmemory() 
            
            # if the current player is significantly better than the best_player replace the best player
            # the replacement is made by just copying the weights of current_player's nn to best_player's nn
            if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
                # if current_NN won send message
                conn.send(((current_player_version, best_player_version), str(scores)))
                
                best_player_version = best_player_version + 1
                best_NN.model.set_weights(current_NN.model.get_weights())
                best_NN.write(env.name, best_player_version)
                
            if len(test_memories.ltmemory) == test_memories.MEMORY_SIZE and current_player_version % 5 == 0:
                pickle.dump(memories, open(run_folder + "memory/test_memory" + str(current_player_version).zfill(4) + ".p", "wb"))

                #print("Evaluating performance of current_NN")
                #current_player.evaluate_accuracy(test_memories.ltmemory)
                #print('\n')
        else:
            time.sleep(10)
        
        
