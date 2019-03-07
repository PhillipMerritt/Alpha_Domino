import numpy as np
import logging
import globals
from copy import deepcopy


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

        self.currentPlayer = 1  # either 1 or -1

        hands, head, queue, public = self._generate_board(self.currentPlayer)  # generate a new board w/ player 2 going first

        # self.currentPlayer = -self.currentPlayer

        self.gameState = GameState(hands, head, queue, public, self.currentPlayer)  # create a GameState
        self.actionSpace = np.zeros(  # action space is a 1x28 array
            (28), dtype=np.int)
        self.grid_shape = (4, 28)  # grid shape is 4x28
        self.input_shape = (4, 28)  # input shape for the neural network is 4x28
        self.name = 'one_armed_joe'
        self.state_size = len(self.gameState.binary)  # size of the entire game state
        self.action_size = len(self.actionSpace)  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        hands, head, queue, public = self._generate_board(1)

        self.gameState = GameState(hands, head, queue, public, 1)

        self.currentPlayer = 1
        return self.gameState

    def _generate_board(self, turn):
        hands = [[],[]]
        queue = globals.queue_reset()   # reset and shuffle the queue

        for i in range(3):
            hands[0].append(queue.pop())  # pop 3 doms of the queue for each player's hand
            hands[1].append(queue.pop())

        dub_found = False  # here we are checking to see if the first player has a double in hand

        for j in hands[1]:
            if j in self.head_values:  # if a double is found play that on the board and remove it from the players hand
                head = self.head_values[j]
                hands[1].remove(j)
                public = [j]
                dub_found = True
                break

        if not dub_found:  # if not choose a random domino from their hand and play that
            choice = np.random.choice(hands[1])
            hands[1].remove(choice)

            public = [choice]

            dom = self.all_domino[
                choice]  # converts the index of the randomly chosen domino to a tuple of its pip values

            choice = np.random.choice(dom)  # choose 1 of the two pip values randomly

            head = choice  # put that domino on the board


        return hands, head, queue, public

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
        currentHead = state.head
        currentPublic = state.public
        currentAV = actionValues

        identities.append((GameState(currentHands,  currentHead, currentQueue, currentPublic, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, hands, head, queue, public, playerTurn):
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
        self.head = head
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

    def _allowedActions(self):

        if np.count_nonzero(self.hands[0]) == 0:  # this shouldn't happen but return no actions if hand is empty
            return []

        while 1:  # checks for actions. If none found draw a domino and try again
            actions = []

            for index in self.hands[0]:
                domino = self.all_domino[np.int(index)]  # convert each index in the hand to a domino tuple
                if self.head == domino[0] or self.head == domino[1]:  # if the head pip value is in the dom
                    actions.append(int(index))  # append that index to actions

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

        return GameState(new_hands, self.head, unknown, self.public, self.playerTurn)




    def _binary(self):  # converts the state to a 4x28 binary representation

        position = np.zeros((4, 28), dtype=np.int)

        for dom in self.hands[0]:
            position[0][dom] = 1
        for dom in self.hands[1]:
            position[1][dom] = 1

        position[2][self.head_indices[self.head]] = 1

        for dom in self.queue:
            position[3][dom] = 1

        return (position)
    # Creates a string id for the state that looks like: "sorted current player's hand | size of opponent's hand | head value | sorted public dominoes"
    # e.x. "2923|4|2|01512161927"
    # the two lists are getting sorted so that if the same game state is reached from different starting points the state will be represented the same
    def _convertStateToId(self):
        hand_copy = sorted(deepcopy(self.hands[0]))  # create a sorted copy of the current player's hand

        id = ''.join(map(str, hand_copy))   # create a string version of the hand

        id += '|' + str(len(self.hands[1])) # add the delimiter and the size of the opponent's hand

        id += '|' + str(self.head)  # add the delimiter and the head value

        public_copy = sorted(deepcopy(self.public)) # create a sorted copy of the list of public dominoes

        id += '|' + ''.join(map(str, public_copy))  # add the delimiter and the public dominoes in string form

        return id

    def _checkForEndGame(self):  # returns 1 if the last player played their last domino or if the current player has no possible plays otherwise returns 0
        if np.count_nonzero(self.hands[1]) == 0 or len(self.allowedActions) == 0:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if np.count_nonzero(self.hands[1]) == 0:
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

    # matches one of the pip values of the chosen action to the current head_domino and replaces the head_domino with another with the value of the opposite pip then returns the new board
    def _convert_head(self, action):
        played_domino = self.all_domino[action]  # get pip tuple of the chosen action

        if played_domino[0] == self.head:  # pip 1 of played domino matches head so pip 2 will be new head
            head = played_domino[1]
        else:  # pip 2 of played domino matches head so pip 1 will be new head
            head = played_domino[0]

        return head

    # creates a copy of the current board with the players hands swapped, makes the chosen action, creates a new gameState
    # then returns the new gameState as well as it's value and an indication of the game being over or not
    def takeAction(self, action):
        new_hands = [[],[]]

        for dom in self.hands[1]:   # copy the opponents hand the current players hand for the new gamestate
            new_hands[0].append(dom)

        for dom in self.hands[0]:   # copy the current players hand to the opponents hand (except for the dom that just got played)
            if dom != action:
                new_hands[1].append(dom)

        self.public.append(action)

        if action != -1:
            head = self._convert_head(action)  # plays action to the board
        else:
            head = self.head

        newState = GameState(new_hands, head, self.queue, self.public, -self.playerTurn)  # create new state

        value = 0
        done = 0

        if newState.isEndGame:  # if the game is over in the new state store its value to value and update done to 1
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self,
               logger):  # this logs each gamestate to a logfile in the run folder. The commented sections will print the game states to the terminal if you uncomment them
        logger.info("Current Turn: {0}".format(self.playerTurn))
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

        logger.info("Head Domino Value: {0}".format(self.head))

        logger.info("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.queue)))
        # print("Head Domino Value: {0}".format(head_domino))
        doms.clear()

        for i in self.allowedActions:
            doms.append(self.all_domino[i])

        logger.info("Available actions: {0}".format(doms))

        if self.p1_val > 0 or self.p2_val >0:
            logger.info("Player {0} pip total - {1}".format(self.playerTurn, self.p1_val))
            logger.info("Player {0} pip total - {1}".format(-self.playerTurn, self.p2_val))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        logger.info('--------------')
        # print('--------------')