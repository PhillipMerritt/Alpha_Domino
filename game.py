import numpy as np
import logging
import globals
from config import PLAYER_COUNT
from copy import deepcopy

# MEXICAN TRAIN
# Game generates GameStates and handles switching between player turns
class Game:

    def __init__(self):     #TODO: add a # of players parameter
        # all_domino is a list of tuples containing the pip value for each domino
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.head_values = {0: 0, 2: 1, 5: 2, 9: 3, 14: 4, 20: 5,
                            27: 6}  # head_values is a dict with the head_indices (doubles) as keys and the corresponding head values as values
        self.head_indices = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14, 5: 20, 6: 27}  # head_indices is the opposite
        # these 2 dicts and the list allow for quick conversions

        hands, trains, queue = self._generate_board()  # generate a new board and choose the starting player based on who has the highest double

        self.gameState = GameState(hands, trains, queue, self.currentPlayer)  # create a GameState
        self.actionSpace = np.zeros(  # action space is 28 * each train
            (28 * len(trains)), dtype=np.int)
        #self.grid_shape = (4, 28)  # grid shape is 7x28
        self.input_shape = (self.gameState.binary.shape)  # input shape for the neural network is the shape of the binary state representation
        self.name = 'mexican_train'
        #self.state_size = len(self.gameState.binary)  # size of the entire game state
        self.action_size = len(self.actionSpace)  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        hands, trains, queue = self._generate_board()

        self.gameState = GameState(hands, trains, queue, self.currentPlayer)

        return self.gameState

    # deal 3 dominoes to each player then choose the starting player based on who has the highest double
    def _generate_board(self):
        highest_double = None

        while not highest_double:
            hands = [[] for players in range(PLAYER_COUNT)]
            queue = globals.queue_reset()   # reset and shuffle the queue

            for i in range(3):
                for p in range(PLAYER_COUNT):
                    hands[p].append(queue.pop())  # pop 3 doms of the queue for each player's hand
            
            # finds the player with the highest double in their hand
            # this will be the starting domino in the hub
            # if no doubles are found highest_double will be None and
            # the hands are dealt again
            highest_double, self.currentPlayer, hands = self.highest_double(hands)
        
        trains = []

        # create a train for each player
        for i in range(PLAYER_COUNT):
            trains.append(Train(highest_double))

        # then add the mexican train
        trains.append(Train(highest_double, True))

        return hands, trains, queue

    # compares the highest double in each player's hand and the player that wins the comparison gets to play their double to the board
    # if no player has a double both hands are redrawn
    def highest_double(self, hands):
        highest_doubles = [-1 for hand in hands]
        doubles = self.head_values.keys()

        for i, hand in enumerate(hands):
            for dom in hand:
                if dom in doubles and dom > highest_doubles[i]:
                    highest_doubles[i] = dom

        winning_double = max(highest_doubles)

        if winning_double == -1:    # if no double was found return None
            return None, None, None
        
        first_player = np.argmax(highest_doubles)   

        hands[first_player].remove(winning_double)  # the highest double will be played to the board

        first_player = (first_player + 1) % PLAYER_COUNT   # first player will be just after the player w/ the highest dom

        return winning_double, first_player, hands






    # once an action has been chosen this function is called and it keeps making actions until their is a choice to be made
    def step(self, action, logger):  # game state makes the chosen action and returns the next state, the value of the next state for the active player in that state, and whether or not the game is over
        while 1:
            next_state, value, done = self.gameState.takeAction(
                action)  # value is always 0 unless the game is over. Otherwise it is -1 since the last player made a winning move
            self.gameState = next_state  # updates the gameState
            self.currentPlayer = -self.currentPlayer  # swaps current player
            info = None  # idk what this is

            self.gameState.render(logger)   # I moved rendering to here so that the automated turns would still be logged

            if done or len(self.gameState.allowedActions) > 1:  # if the game is over or the current player has a choice break the loop
                break
            elif len(self.gameState.allowedActions) == 1:   # else takeAction() with the one action available
                action = self.gameState.allowedActions[0]
            else:                                           # or if no actions are available pass turn by taking action -1
                action = -1

        return ((next_state, value, done, info))

    def identities(self, state, actionValues):  # TODO: take out the uneccesary stuff here
        identities = [(state, actionValues)]

        currentHands = state.hands
        currentQueue = state.queue
        currentTrains = state.trains
        currentPublic = state.public
        currentAV = actionValues

        identities.append((GameState(currentHands,  currentTrains, currentQueue, currentPublic, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, hands, trains, queue, playerTurn, passed = [False for player in PLAYER_COUNT]):
        # all_domino is a list of tuples containing the pip value for each domino
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.head_values = {0: 0, 2: 1, 5: 2, 9: 3, 14: 4, 20: 5,
                            27: 6}  # head_values is a dict with the head_indices (doubles) as keys and the corresponding head values as values
        self.head_indices = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14, 5: 20, 6: 27}  # head_indices is the opposite
        # these 2 dicts and the list allow for quick conversions

        self.p1_val = 0  # scores from a block ending
        self.p2_val = 0

        self.hands = hands
        self.trains = trains
        self.queue = queue
        self.public = public

        self.playerTurn = playerTurn
        self.drawCount = 0  # tracks the # of times this player has drawn this turn. only used for logging
        self.public_id = self.get_public_info()
        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now

        self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move
        self.score = self._getScore()
        self.passed = passed

    def _draw(self):  # draws a random domino then updates binary and id. If there are no more dominos to draw return false

        if len(self.queue) > 0:  # if there are dominoes to draw
            self.drawCount += 1

            self.hands[0].append(self.queue.pop())  # randomly pop one from the boneyard and place it in the players hand

            self.binary = self._binary()
            self.id = self._convertStateToId()

            return True

        return False

    # generates a list of all allowed actions. If there are no available actions dominoes are drawn if available
    # until there are actions to be made. The actions are in the form of (train,domino) (ex. domino 14 to train 3 would be (3, 14))
    # actually maybe it will be action = (train num * 28) + action (ex. domino 14 to train 3 would be (3*28)+14 = 98
    def _allowedActions(self):
        while 1:  # checks for actions. If none found draw a domino and try again
            actions = []

            for index in self.hands[0]: # for each domino in hand
                domino = self.all_domino[np.int(index)]  # convert each index in the hand to a domino tuple
                for head in np.nonzero(self.trains)[0]:  # for the head value of each train

                    if head == domino[0]:  # if the head pip value is in the dom
                        actions.append(index)  # append that index to actions
                    elif head == domino[1]:
                        actions.append(28+index)

            if len(actions) > 0:  # if there are any available actions return them
                return actions
            elif not self._draw():  # if no actions found draw a domino
                self.passed = True
                return []  # if drawing a domino fails return an empty list

    # creates a list of hidden information by adding the opponent's hand back into the queue
    # then generate a cloned gameState with the opponents hand generated from the shuffled
    # unknown list
    def CloneAndRandomize(self):
        unknown = deepcopy(self.queue)  # create a deep copy of the queue

        for i in range(1, PLAYER_COUNT):
            for dom in self.hands[(self.playerTurn + i) % PLAYER_COUNT]: # put all of the opponent's dominoes in with the rest of the unknown dominoes
                unknown.append(dom)

        new_hands = [[] for player in range(PLAYER_COUNT)]

        for dom in self.hands[self.playerTurn]:   # copy over the current players hand
            new_hands[self.playerTurn].append(dom)

        np.random.shuffle(unknown)

        for i in range(1, PLAYER_COUNT):
            for k in range(len(self.hands[(self.playerTurn + i) % PLAYER_COUNT])):
                new_hands[1].append(unknown.pop())

        return GameState(new_hands, self.trains, unknown, self.playerTurn)



    # converts the state to a (2 * player_count + 3)x28 binary representation 
    # (current_player's hand, size of each other player's hand, each player's train, mexican train, marked train indices, available heads to play on)
    def _binary(self):  # TODO signify multiples of a single head value being available

        state = np.zeros((2 * PLAYER_COUNT + 2, 28), dtype=np.int)
        state[0][self.hands[self.playerTurn]] = 1   # current player's hand
        for i in range(1, PLAYER_COUNT):
            state[i][len(self.hands[(self.playerTurn + i) % PLAYER_COUNT])] = 1 # length of each other player's hand

        for i in range(PLAYER_COUNT): # each train
            state[i + PLAYER_COUNT] = self.trains[(self.playerTurn + i) % PLAYER_COUNT].get_binary()
        
        state[2*PLAYER_COUNT] = self.trains[PLAYER_COUNT].get_binary()

        for i in range(PLAYER_COUNT):
            index = (self.playerTurn + i) % PLAYER_COUNT
            if self.trains[index].marked:
                state[2 * PLAYER_COUNT + 1][index] = 1  # train marked
                state[2 * PLAYER_COUNT + 2][self.trains[index].head] = 1 # available head to play on
            elif i == 0:
                state[2 * PLAYER_COUNT + 2][self.trains[index].head] = 1 # available head to play on
        
        state[2 * PLAYER_COUNT + 1][PLAYER_COUNT] = 1  # mexican train marked
        state[2 * PLAYER_COUNT + 2][self.trains[PLAYER_COUNT].head] = 1 # available head to play on

        return state



        
    # Creates a string id for the state which is used to identify nodes in the ISMCTS
    def _convertStateToId(self):
        id = self.public_id + str(sorted(self.hands[self.playerTurn])) # current player's hand appended to public info

        return id

    def get_public_info(self):
        public_id = ''
        for i in range(PLAYER_COUNT):
            public_id += '|' + len(self.hands[i])
        for train in self.trains:
            public_id += '|' + train.get_string()
        
        return public_id

    def _checkForEndGame(self):  # returns 1 if any player has an empty hand else 0 or all players have passed
        for hand in self.hands:
            if len(hand) == 0:
                return 1
        if False not in self.passed:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose

        if self.isEndGame:
            # each player has ran out of dominoes so their tiles are flipped and the pips are added up
            # the player with the lowest total wins
            if False not in self.passed:
                totals = [sum([sum(self.all_domino[dom]) for dom in hand]) for hand in self.hands]
                winner = np.argmin(totals)
                if winner != type(int): # pick a random winner if there is a tie
                    winner = np.random.choice(winner)
                else:
                    temp = []
                    for i in range(PLAYER_COUNT):
                        if i == winner:
                            temp.append(1)
                        else:
                            temp.append(-1)
            else:
                temp = []
                for hand in self.hands:
                    if len(hand) == 0:
                        temp.append(1)
                    else:
                        temp.append(-1)
                
                return temp
        else:
            return [0 for player in range(PLAYER_COUNT)]

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    # matches one of the pip values of the chosen action to the head value of the chosen train and replaces the head_domino with the value of the opposite pip then returns the new trains
    def _convert_head(self, action):
        played_domino = self.all_domino[action%28]  # get pip tuple of the chosen action

        dub = False
        if played_domino[0] == played_domino[1]:
            dub = True

        trains = deepcopy(self.trains)

        tuple_index = int(action/28)

        if dub:
            trains[played_domino[tuple_index]] += 1
        else:
            trains[played_domino[tuple_index]] += 1
            trains[played_domino[(tuple_index+1)%2]] -= 1

        return trains

    # creates a copy of the current board with the players hands swapped, makes the chosen action, creates a new gameState
    # then returns the new gameState as well as it's value and an indication of the game being over or not
    def takeAction(self, action):
        new_hands = [[],[]]

        for dom in self.hands[1]:   # copy the opponents hand the current players hand for the new gamestate
            new_hands[0].append(dom)

        for dom in self.hands[0]:   # copy the current players hand to the opponents hand (except for the dom that just got played)
            if dom != (action%28):
                new_hands[1].append(dom)

        public = deepcopy(self.public)

        if action != -1:
            trains = self._convert_head(action)  # plays action to the board
            public.append(action%28)
        else:
            trains = deepcopy(self.trains)

        newState = GameState(new_hands, trains, self.queue, (self.playerTurn + 1) % len(new_hands))  # create new state

        value = 0
        done = 0

        if newState.isEndGame:  # if the game is over in the new state store its value to value and update done to 1
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self, logger):  # this logs each gamestate to a logfile in the run folder. The commented sections will print the game states to the terminal if you uncomment them
        if self.playerTurn == 1:
            logger.info("Current Turn: {0}".format(1))
        else:
            logger.info("Current Turn: {0}".format(2))
        # print("Current Turn: {0}".format(self.playerTurn))
        logger.info("# of draws this turn: {0}".format(self.drawCount))
        # print("# of draws this turn: {0}".format(self.drawCount))

        if self.playerTurn == 1:  # added this so the turn isn't switching back and forth in logging
            turn = 0
            o_turn = 1
        else:
            turn = 1
            o_turn = 0

        doms = []

        for i in self.hands[turn]:
            doms.append(self.all_domino[np.int(i)])

        if len(doms) > 0:
            logger.info("Player 1 hand: {0}".format(doms))  # p1 hand
            # print("Player 1 hand: {0}".format(doms))
        else:
            logger.info("Player 1 hand: Empty")
            # print("Player 1 hand: Empty")

        doms.clear()

        for i in self.hands[o_turn]:
            doms.append(self.all_domino[np.int(i)])

        if len(doms) > 0:
            logger.info("Player 2 hand: {0}".format(doms))  # p1 hand
            # print("Player 2 hand: {0}".format(doms))
        else:
            logger.info("Player 2 hand: Empty")
            # print("Player 2 hand: Empty")

        logger.info("Trains (value: #): 0:{0} 1:{1} 2:{2} 3:{3} 4:{4} 5:{5} 6:{6}".format(self.trains[0],self.trains[1],self.trains[2],self.trains[3],self.trains[4],self.trains[5],self.trains[6]))

        logger.info("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.queue)))
        # print("Head Domino Value: {0}".format(head_domino))
        doms.clear()

        actions = []
        for i in self.allowedActions:
            doms.append(self.all_domino[i%28])

        logger.info("Available actions (action #, train #, domino): {0}".format(doms))

        if self.p1_val > 0 or self.p2_val >0:
            logger.info("Player {0} pip total - {1}".format(self.playerTurn, self.p1_val))
            logger.info("Player {0} pip total - {1}".format(-self.playerTurn, self.p2_val))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        logger.info('--------------')
        # print('--------------')

class Train:
    def __init__(self, first_dom, marked=False):
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.head_values = {0: 0, 2: 1, 5: 2, 9: 3, 14: 4, 20: 5,
                            27: 6}
        self.head_indices = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14, 5: 20, 6: 27}
        self.doms = [first_dom]
        self.head = self.head[first_dom]
        self.marked = marked

    def add(self, dom):
        self.doms.append(dom)
        tup = self.all_domino[dom]
        if tup[0] == self.head:
            self.head = tup[1]
        else:
            self.head = tup[0]

    def match(self, dom):
        tup = self.all_domino[dom]
        if self.head in tup:
            return True
        return False
    
    def mark(self):
        self.marked = True
    
    def unmark(self):
        self.marked = False
    
    def get_binary(self):
        b = np.zeros(28, dtype = np.int)
        b[self.doms] = 1
        return b
    
    def get_string(self):
        sorted_doms = sorted(self.doms)
        return str(sorted_doms) + str(self.marked)
