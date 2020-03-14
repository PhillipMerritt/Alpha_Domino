import numpy as np
from copy import deepcopy
from collections import defaultdict

global unshuffled_queue
unshuffled_queue = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

global shuffled_queue
shuffled_queue = []

global INDEX2TUP
INDEX2TUP = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]

global TUP2INDICES
TUP2INDICES = [[0, 1, 3, 6, 10, 15, 21], [1, 2, 4, 7, 11, 16, 22], [3, 4, 5, 8, 12, 17, 23], [6, 7, 8, 9, 13, 18, 24], [10, 11, 12, 13, 14, 19, 25], [15, 16, 17, 18, 19, 20, 26], [21, 22, 23, 24, 25, 26, 27]]

def queue_reset():
    shuffled_queue = deepcopy(unshuffled_queue)

    #np.random.seed()

    np.random.shuffle(shuffled_queue)

    return shuffled_queue