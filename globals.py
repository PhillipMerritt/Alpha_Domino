import numpy as np
from collections import defaultdict
from config import PLAYER_COUNT, TEAM_SIZE

global shuffled_queue
shuffled_queue = []

global HANDSIZE
global MAX_PIP

if TEAM_SIZE > 1:
    MAX_PIP = 6
    HANDSIZE = 7
else:
    if PLAYER_COUNT < 4:
        MAX_PIP = 9
        HANDSIZE = 8
    elif PLAYER_COUNT < 7:
        MAX_PIP = 12
        HANDSIZE = 12
    else:
        MAX_PIP = 15
        HANDSIZE = 10
    
dubs = []
tups = []
for high in range(MAX_PIP + 1):
    for low in range(high + 1):
        tups.append((low, high))

global INDEX2TUP
INDEX2TUP = tups

global DOM_COUNT
DOM_COUNT = len(tups)

global unshuffled_queue
unshuffled_queue = list(range(DOM_COUNT))

values = []
indices = [[] for i in range(MAX_PIP + 1)]
for i, (low, high) in enumerate(INDEX2TUP):
    indices[low].append(i)

    if low != high:
        indices[high].append(i)
    else:
        dubs.append(i)
        values.append(low)

"""for i, a in enumerate(indices):
    print('{0}: {1}'.format(i, a))

quit(0)"""

global DOUBLES
DOUBLES = dubs

tempa = {}
tempb = {}
for i, dub in enumerate(dubs):
    tempa.update({dub: values[i]})
    tempb.update({values[i]: dub})

global HEAD_VALUES
HEAD_VALUES = tempa.copy()

global HEAD_INDICES
HEAD_INDICES = tempb.copy()

global PIP2INDICES
PIP2INDICES = [set(i) for i in indices]

def queue_reset():
    shuffled_queue = list(unshuffled_queue)

    #np.random.seed()

    np.random.shuffle(shuffled_queue)

    return shuffled_queue