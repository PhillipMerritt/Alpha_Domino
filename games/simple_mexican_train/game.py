import numpy as np
import logging
import globals
from copy import deepcopy

# MEXICAN TRAIN
# Game generates GameStates and handles switching between player turns
class Game:

    def __init__(self):
        # all_domino is a list of tuples containing the pip value for each domino
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.head_values = {0: 0, 2: 1, 5: 2, 9: 3, 14: 4, 20: 5,
                            27: 6}  # head_values is a dict with the head_indices (doubles) as keys and the corresponding head values as values
        self.head_indices = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14, 5: 20, 6: 27}  # head_indices is the opposite
        # these 2 dicts and the list allow for quick conversions

        self.player_count = 4

        hands, trains, queue, public = self._generate_board()  # generate a new board and choose the starting player based on who has the highest double

        self.gameState = GameState(hands, trains, queue, public, self.currentPlayer)  # create a GameState
        self.actionSpace = np.zeros(  # action space is a 4x28 array
            (56), dtype=np.int)
        self.grid_shape = (4, 28)  # grid shape is 7x28
        self.input_shape = (4, 28)  # input shape for the neural network is 7x28
        self.name = 'simple_mexican_train'
        self.state_size = len(self.gameState.binary)  # size of the entire game state
        self.action_size = len(self.actionSpace)  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        hands, trains, queue, public = self._generate_board()

        self.gameState = GameState(hands, trains, queue, public, self.currentPlayer)

        return self.gameState

    # deal 3 dominoes to each player then choose the starting player based on who has the highest double
    def _generate_board(self):
        hands = [[],[]]
        public = []
        queue = globals.queue_reset()   # reset and shuffle the queue

        for i in range(3):
            hands[0].append(queue.pop())  # pop 3 doms of the queue for each player's hand
            hands[1].append(queue.pop())

        # highest_double returns the results of comparing doubles
        # currentPlayer is set to the player that didn't have the highest double
        hands, self.currentPlayer, head_index, head = self.highest_double(hands, queue)

        if self.currentPlayer == -1:    # if player 1 got to go first swap the hands
            temp = hands[0]
            hands[0] = hands[1]
            hands[1] = temp

        public.append(head_index)   # add the played double to the list of public dominoes

        trains = np.zeros((28), dtype=np.int)  # the starting domino can have 4 trains coming off of it so set all 4 to the same value
        trains[head_index] = 4

        return hands, trains, queue, public

    # compares the highest double in each player's hand and the player that wins the comparison gets to play their double to the board
    # if neither player has a double both draw and compare again.
    def highest_double(self, hands, queue):
        while 1:
            highest_p1_value = -1
            highest_p1_index = -1
            highest_p2_value = -1
            highest_p2_index = -1

            for i in hands[0]:
                if i in self.head_values:
                    val = self.head_values[i]

                    if val > highest_p1_value:
                        highest_p1_value = val
                        highest_p1_index = i

            for i in hands[1]:
                if i in self.head_values:
                    val = self.head_values[i]

                    if val > highest_p2_value:
                        highest_p2_value = val
                        highest_p2_index = i

            if highest_p1_value > highest_p2_value:
                hands[0].remove(highest_p1_index)
                return hands, -1, highest_p1_index, highest_p1_value
            elif highest_p1_value < highest_p2_value:
                hands[1].remove(highest_p2_index)
                return hands, 1, highest_p2_index, highest_p2_value

            hands[0].append(queue.pop())
            hands[1].append(queue.pop())


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

    def identities(self, state, actionValues):  # haven't looked into what this function is doing quite yet
        identities = [(state, actionValues)]

        currentHands = state.hands
        currentQueue = state.queue
        currentTrains = state.trains
        currentPublic = state.public
        currentAV = actionValues

        identities.append((GameState(currentHands,  currentTrains, currentQueue, currentPublic, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, hands, trains, queue, public, playerTurn):
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
        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now

        self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move
        self.score = self._getScore()
        self.passed = False

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

        for dom in self.hands[1]: # put all of the opponent's dominoes in with the rest of the unknown dominoes
            unknown.append(dom)

        new_hands = [[],[]]

        for dom in self.hands[0]:   # copy over the current players hand
            new_hands[0].append(dom)

        np.random.shuffle(unknown)

        for i in range(len(self.hands[1])):
            new_hands[1].append(unknown.pop())

        return GameState(new_hands, self.trains, unknown, deepcopy(self.public), self.playerTurn)




    def _binary(self):  # converts the state to a 7x28 binary representation (hand, hand, boneyard, train 1, train 2, train 3, train 4)

        position = np.zeros((4, 28), dtype=np.int)

        for dom in self.hands[0]:
            position[0][dom] = 1
        for dom in self.hands[1]:
            position[1][dom] = 1

        for dom in self.queue:
            position[2][dom] = 1

        for i in range(6):
            position[3][i] = self.trains[i]

        return (position)
    # Creates a string id for the state that looks like: "sorted current player's hand | size of opponent's hand | sorted public dominoes | train head 1 | train head 2 | train head 3 | train head 4"
    # e.x. "2923|4|2|01512161927"
    # the two lists are getting sorted so that if the same game state is reached from different starting points the state will be represented the same
    def _convertStateToId(self):
        hand_copy = sorted(deepcopy(self.hands[0]))  # create a sorted copy of the current player's hand

        id = ''.join(map(str, hand_copy))   # create a string version of the hand

        id += '|' + str(len(self.hands[1])) # add the delimiter and the size of the opponent's hand

        public_copy = sorted(deepcopy(self.public))  # create a sorted copy of the list of public dominoes

        id += '|' + ''.join(map(str, public_copy))  # add the delimiter and the public dominoes in string form

        id += '|' + ''.join(map(str, self.trains))

        return id

    def _checkForEndGame(self):  # returns 1 if the last player played their last domino or if the current player has no possible plays otherwise returns 0
        if np.count_nonzero(self.hands[1]) == 0 or len(self.allowedActions) == 0:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if len(self.hands[1]) == 0:
            return (-1, -1, 1)

        # both players have ran out of dominoes so their tiles are flipped and the pips are added up
        # the player with the lowest total wins
        if self.isEndGame:
            for index in self.hands[0]:
                domino = self.all_domino[np.int(index)]  # convert each index in the hand to a domino tuple
                self.p1_val += domino[0] + domino[1]

            for index in self.hands[1]:
                domino = self.all_domino[np.int(index)]  # convert each index in the hand to a domino tuple
                self.p2_val += domino[0] + domino[1]

            if self.p1_val < self.p2_val:
                return (1, 1, -1)
            elif self.p1_val > self.p2_val:
                return (-1, -1, 1)

        return (0, 0, 0)

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

        newState = GameState(new_hands, trains, self.queue, public, -self.playerTurn)  # create new state

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