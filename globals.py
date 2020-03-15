import numpy as np
from copy import deepcopy
from collections import defaultdict
from config import PLAYER_COUNT

global shuffled_queue
shuffled_queue = []

global HANDSIZE
if PLAYER_COUNT < 4:
    max_pip = 9
    HANDSIZE = 8
elif PLAYER_COUNT < 7:
    max_pip = 12
    HANDSIZE = 12
else:
    max_pip = 15
    HANDSIZE = 10
dubs = []
tups = []
for high in range(max_pip + 1):
    for low in range(high + 1):
        tups.append((low, high))

global INDEX2TUP
INDEX2TUP = tups

global DOM_COUNT
DOM_COUNT = len(tups)

global unshuffled_queue
unshuffled_queue = list(range(DOM_COUNT))

values = []
indices = [[] for i in range(max_pip + 1)]
for i, (low, high) in enumerate(INDEX2TUP):
    indices[low].append(i)

    if low != high:
        indices[high].append(i)
    else:
        dubs.append(i)
        values.append(low)

global DOUBLES
DOUBLES = dubs

tempa = {}
tempb = {}
for i, dub in enumerate(dubs):
    tempa.update({dub: values[i]})
    tempb.update({values[i]: dub})

global HEAD_VALUES
HEAD_VALUES = tempa.copy()
print(HEAD_VALUES)

global HEAD_INDICES
HEAD_INDICES = tempb.copy()

global PIP2INDICES
PIP2INDICES = indices

def queue_reset():
    shuffled_queue = deepcopy(unshuffled_queue)

    #np.random.seed()

    np.random.shuffle(shuffled_queue)

    return shuffled_queue