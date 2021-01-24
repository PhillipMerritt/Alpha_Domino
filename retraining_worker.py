def retraining_worker(conn):
    from game import Game
    import initialise
    import config
    from config import PLAYER_COUNT, TEAM_SIZE, BATCH_SIZE, TRAINING_LOOPS
    from model import Residual_CNN, import_tf
    import_tf(1024 * 2)
    import numpy as np
    import time
    
    env = Game()
    
    ######## LOAD MODEL IF NECESSARY ########

    # create an untrained neural network objects from the config file
    if len(env.grid_shape) == 2:
        shape = (1,) + env.grid_shape
    else:
        shape = env.grid_shape

    if TEAM_SIZE > 1:
        current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, int(PLAYER_COUNT / TEAM_SIZE),
                            config.HIDDEN_CNN_LAYERS)
    else:
        current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, shape, PLAYER_COUNT,
                            config.HIDDEN_CNN_LAYERS)
    
    # If loading an existing neural netwrok, set the weights from that model
    if initialise.INITIAL_MODEL_VERSION != None:
        m_tmp = current_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, initialise.INITIAL_MODEL_VERSION)
        current_NN.model.set_weights(m_tmp.get_weights())
        
    train_overall_loss = []
        
    while 1:
        # request memory samples
        conn.send((TRAINING_LOOPS, BATCH_SIZE))
        
        # wait for memory samples
        conn.poll(None)
        data = conn.recv()
        
        if data:
            # train on sampled memories
            for i, minibatch in enumerate(data):
                training_states = np.array([current_NN.convertToModelInput(row['state']) for row in minibatch])
                training_targets = {'value_head': np.array([row['value'] for row in minibatch])}
                
                fit = current_NN.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1,\
                                        validation_split=0, batch_size=32)
                
                if i == 0:
                    init_loss = fit.history['loss'][0]
                
                train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            
            """display.clear_output(wait=True)
            display.display(pl.gcf())
            pl.gcf().clear()
            time.sleep(.25)

            print('\n')
            current_NN.printWeightAverages()

            print("Max = {0}, Min = {1}, latest = {2}".format(max(self.train_overall_loss), min(self.train_overall_loss), self.train_overall_loss[-1]))
            print("Loss reduction: {}".format(init_loss - fit.history['loss'][0]))"""
            
            # send new current_NN weights
            conn.send((current_NN.model.get_weights(), train_overall_loss[-1]))
        else:
            time.sleep(10)
        
