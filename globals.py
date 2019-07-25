import numpy as np
from copy import deepcopy

global unshuffled_queue
unshuffled_queue = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

global shuffled_queue
shuffled_queue = []

def queue_reset():
    shuffled_queue = deepcopy(unshuffled_queue)

    np.random.seed()

    np.random.shuffle(shuffled_queue)

    return shuffled_queue