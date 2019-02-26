import numpy as np
import logging


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

        board = self._generate_board(self.currentPlayer)  # generate a new board w/ player 1 going first

        # self.currentPlayer = -self.currentPlayer

        self.gameState = GameState(board, self.currentPlayer)  # create a GameState
        self.actionSpace = np.zeros(  # action space is a 1x28 array
            (28), dtype=np.int)
        self.grid_shape = (4, 28)  # grid shape is 4x28
        self.input_shape = (4, 28)  # input shape for the neural network is 4x28
        self.name = 'one_armed_joe'
        self.state_size = len(self.gameState.binary)  # size of the entire game state
        self.action_size = len(self.actionSpace)  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        board = self._generate_board(1)

        self.gameState = GameState(board, 1)

        self.currentPlayer = 1
        return self.gameState

    def _generate_board(self, turn):
        board = np.zeros(  # 0 - player hand, 1 - opponent hand, 2 - board, 3-bonepile
            (4, 28), dtype=np.int)  # the pip value of the head domino is represented as a double domino of that value
        # ex. If the 6:3 domino is the head domino with the 3 as the side that can be played on
        # it is represented as the 3:3 domino in the board list.

        for i in range(28):  # sets all boneyard elements to true
            board[3][i] = 1

        board[3][
            24] = 0  # everything from here to the return statement is generating a specific board state for debugging
        # normally both players would take turns drawing a random domino from the bonepile
        """for i in range(3):
            choices = np.nonzero(board[3])[0]
            choice = np.random.choice(choices)
            board[0][choice] = 1
            board[3][choice] = 0
        for i in range(2):
            choices = np.nonzero(board[3])[0]
            choice = np.random.choice(choices)
            board[1][choice] = 1
            board[3][choice] = 0

        board[2][27] = 1

        return board"""

        for i in range(3):
            choices = np.nonzero(board[3])[0]
            choice = np.random.choice(choices)
            board[0][choice] = 1
            board[3][choice] = 0

            choices = np.nonzero(board[3])[0]
            choice = np.random.choice(choices)
            board[1][choice] = 1
            board[3][choice] = 0

        dub_found = False  # here we are checking to see if the first player has a double in hand
        choices = np.nonzero(board[1])[0]  # creates a list of the non-zero indices from the 1st players hand
        dubs = [0, 2, 5, 9, 14, 20, 27]  # list of all double indices (0:0, 1:1, 2:2, etc...)

        for j in choices:
            if j in dubs:  # if a double is found play that on the board and remove it from the players hand
                board[2][j] = 1
                board[1][j] = 0
                dub_found = True
                break

        if not dub_found:  # if not choose a random domino from their hand and play that
            choices = np.nonzero(board[1])[
                0]  # this is a pretty bad solution for this that should be updated at some point
            choice = np.random.choice(choices)
            board[1][choice] = 0

            dom = self.all_domino[
                choice]  # converts the index of the randomly chosen domino to a tuple of its pip values

            choice = np.random.choice(dom)  # choose 1 of the two pip values randomly

            head_index = self.head_indices[
                choice]  # convert the chosen pip value into the index of the double domino of that value

            board[2][head_index] = 1  # put that domino on the board

        return board

    # once an action has been chosen this function is called
    def step(self, action, logger):  # game state makes the chosen action and returns the next state, the value of the next state for the active player in that state, and whether or not the game is over
        while 1:
            next_state, value, done = self.gameState.takeAction(
                action)  # value is always 0 unless the game is over. Otherwise it is -1 since the last player made a winning move
            self.gameState = next_state  # updates the gameState
            self.currentPlayer = -self.currentPlayer  # swaps current player
            info = None  # idk what this is

            self.gameState.render(logger)

            if done or len(self.gameState.allowedActions) > 1:
                break
            elif len(self.gameState.allowedActions) == 1:
                action = self.gameState.allowedActions[0]
            else:
                action = -1

        return ((next_state, value, done, info))

    def identities(self, state, actionValues):  # haven't looked into what this function is doing quite yet
        identities = [(state, actionValues)]

        currentBoard = state.board
        currentAV = actionValues

        identities.append((GameState(currentBoard, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, board, playerTurn):
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

        self.board = board
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

    def to_domino(self, index):  # returns the pip values of the domino at index

        return self.all_domino[int(index)]

        # there is probably too much redundancy in the following two functions, but it hasn't seemed important enough to fix yet

    def _get_head_value(self):  # gets the index from board[2] and then returns its pip value
        index = np.nonzero(self.board[2])[0][0]

        return self.head_values[index]

    def _head_value(self, index):  # returns the pip value of the double at this index

        return self.head_values[index]

    def head_index(self, value):  # returns the index of the double domino of the value

        return self.head_indices[value]

    def _draw(
            self):  # draws a random domino then updates binary and id. If there are no more dominos to draw return false
        bones = np.nonzero(self.board[3])[0]  # array of dominoes in bonepile

        if len(bones) > 0:  # if there are dominoes to draw
            self.drawCount += 1

            draw = np.random.choice(bones)  # randomly choose one

            self.board[0][draw] = 1  # add to hand
            self.board[3][draw] = 0  # remove from bonepile

            self.binary = self._binary()
            self.id = self._convertStateToId()

            return True

        return False

    def _allowedActions(self):
        head_domino = self._get_head_value()  # get the pip value of head dom

        if np.count_nonzero(self.board[0]) == 0:  # this shouldn't happen but return no actions if hand is empty
            return []

        while 1:  # checks for actions. If none found draw a domino and try again
            hand = np.nonzero(self.board[0])[0]  # get arr of hand indices

            actions = []

            for index in hand:
                domino = self.to_domino(np.int(index))  # convert each index in the hand to a domino tuple
                if head_domino == domino[0] or head_domino == domino[1]:  # if the head pip value is in the dom
                    actions.append(int(index))  # append that index to actions

            if len(actions) > 0:  # if there are any available actions return them
                return actions
            elif not self._draw():  # if no actions found draw a domino
                self.passed = True
                return []  # if drawing a domino fails return an empty list

    def _binary(self):  # just creates a copy of the board and returns it

        position = np.zeros((4, 28), dtype=np.int)

        position[0][self.board[0] == 1] = 1
        position[1][self.board[1] == 1] = 1
        position[2][self.board[2] == 1] = 1
        position[3][self.board[3] == 1] = 1

        return (position)

    def _convertStateToId(
            self):  # appends a list of the nonzero indices from each of the board lists one after the other
        if self.playerTurn == 1:
            position = np.append(self.board[0], self.board[1])
        else:
            position = np.append(self.board[1], self.board[0])

        position = np.append(position, self.board[2])
        position = np.append(position, self.board[3])

        id = ''.join(map(str, position))

        return id

    def _checkForEndGame(self):  # returns 1 if the last player played their last domino or if the current player has no possible plays otherwise returns 0
        if np.count_nonzero(self.board[1]) == 0 or len(self.allowedActions) == 0:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if np.count_nonzero(self.board[1]) == 0:
            return (-1, -1, 1)

        # both players have ran out of dominoes so their tiles are flipped and the pips are added up
        # the player with the lowest total wins
        if self.isEndGame:
            hand = np.nonzero(self.board[0])[0]  # get arr of hand indices

            for index in hand:
                domino = self.to_domino(np.int(index))  # convert each index in the hand to a domino tuple
                self.p1_val += domino[0] + domino[1]

            hand = np.nonzero(self.board[1])[0]  # get arr of op hand indices

            for index in hand:
                domino = self.to_domino(np.int(index))  # convert each index in the hand to a domino tuple
                self.p2_val += domino[0] + domino[1]

            if self.p1_val < self.p2_val:
                return (1, 1, -1)
            elif self.p1_val > self.p2_val:
                return (-1, -1, 1)

        return (0, 0, 0)

    def _getScore(
            self):
        """count_a = np.count_nonzero(self.board[0])
        count_a = 27 - count_a

        count_b = np.count_nonzero(self.board[1])
        count_b = 27 - count_b

        return (count_a, count_b)"""
        tmp = self.value
        return (tmp[1], tmp[2])

    # matches one of the pip values of the chosen action to the current head_domino and replaces the head_domino with another with the value of the opposite pip then returns the new board
    def _convert_head(self, action, newBoard):
        played_domino = self.to_domino(action)  # get pip tuple of the chosen action

        head_domino = self._get_head_value()  # get the pip value of the current head domino

        if played_domino[0] == head_domino:  # pip 1 of played domino matches head so pip 2 will be new head
            newBoard[2][self.head_index(played_domino[1])] = 1
        else:  # pip 2 of played domino matches head so pip 1 will be new head
            newBoard[2][self.head_index(played_domino[0])] = 1

        newBoard[1][action] = 0  # remove played domino from player's hand

        return newBoard

    # creates a copy of the current board with the players hands swapped, makes the chosen action, creates a new gameState
    # then returns the new gameState as well as it's value and an indication of the game being over or not
    def takeAction(self, action):
        newBoard = np.zeros(  # 0 - next player's hand, 1 - current player's hand, 2 - board, 3-bonepile
            (4, 28), dtype=np.int)
        newBoard[0][self.board[1] == 1] = 1  # hands are swapped for next game state
        newBoard[1][self.board[0] == 1] = 1
        newBoard[3][self.board[3] == 1] = 1

        if action != -1:
            newBoard = self._convert_head(action, newBoard)  # plays action to the board
        else:
            newBoard[2][self.board[2] == 1] = 1

        newState = GameState(newBoard, -self.playerTurn)  # create new state

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

        hand_indices = np.nonzero(self.board[turn])[0]
        for i in hand_indices:
            doms.append(self.to_domino(np.int(i)))

        if len(doms) > 0:
            logger.info("Player 1 hand: {0}".format(doms))  # p1 hand
            # print("Player 1 hand: {0}".format(doms))
        else:
            logger.info("Player 1 hand: Empty")
            # print("Player 1 hand: Empty")

        doms.clear()

        hand_indices = np.nonzero(self.board[o_turn])[0]
        for i in hand_indices:
            doms.append(self.to_domino(np.int(i)))

        if len(doms) > 0:
            logger.info("Player 2 hand: {0}".format(doms))  # p1 hand
            # print("Player 2 hand: {0}".format(doms))
        else:
            logger.info("Player 2 hand: Empty")
            # print("Player 2 hand: Empty")

        board_indices = np.nonzero(self.board[2])[0]

        if len(board_indices) == 0:
            head_domino = -1
        else:
            head_domino = self._head_value(board_indices[0])

        logger.info("Head Domino Value: {0}".format(head_domino))

        logger.info("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))
        # print("Head Domino Value: {0}".format(head_domino))
        doms.clear()

        for i in self.allowedActions:
            doms.append(self.to_domino(i))

        logger.info("Available actions: {0}".format(doms))

        if self.p1_val > 0 or self.p2_val >0:
            logger.info("Player {0} pip total - {1}".format(self.playerTurn, self.p1_val))
            logger.info("Player {0} pip total - {1}".format(-self.playerTurn, self.p2_val))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        logger.info('--------------')
        # print('--------------')