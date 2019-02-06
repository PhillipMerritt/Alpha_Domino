import numpy as np
import logging

class Game:

    def __init__(self):
        self.currentPlayer = 1



        board = self._generate_board(self.currentPlayer)

        self.currentPlayer = -self.currentPlayer

        self.gameState = GameState(board, self.currentPlayer)
        self.actionSpace = np.zeros(
            (28), dtype=np.int)
        self.grid_shape = (4, 28)
        self.input_shape = (4, 28)
        self.name = 'one_armed_joe'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)
        #print("state_size - ", self.state_size)
        #print("action_size - ", self.action_size)

    def reset(self):
        board = self._generate_board(1)

        self.currentPlayer = -1

        self.gameState = GameState(board, self.currentPlayer)

        #self.currentPlayer = 1
        return self.gameState

    def _generate_board(self, turn):
        board = np.zeros(  # 0 - player hand, 1 - opponent hand, 2 - board, 3-bonepile
            (4, 28), dtype=np.int)

        for i in range(28):
            board[3][i] = 1

        for i in range(3):
            choices = np.nonzero(board[3])[0]
            choice = np.random.choice(choices)
            board[0][choice] = 1
            board[3][choice] = 0

            choices = np.nonzero(board[3])[0]
            choice = np.random.choice(choices)
            board[1][choice] = 1
            board[3][choice] = 0

        dub_found = False
        choices = np.nonzero(board[0])[0]
        dubs = [0, 2, 5, 9, 14, 20, 27]

        for j in choices:
            if j in dubs:
                board[2][j] = 1
                board[0][j] = 0
                dub_found = True
                break

        if not dub_found:
            if turn == 1:
                choices = np.nonzero(board[0])[0]
                choice = np.random.choice(choices)
                board[0][choice] = 0
            else:
                choices = np.nonzero(board[1])[0]
                choice = np.random.choice(choices)
                board[1][choice] = 0

            choice += 1

            i = .5 * (np.sqrt((8 * (choice)) + 1) - 1)

            i = np.floor(i)
            k = choice - ((i * (i + 1)) / 2)

            dom = (int(k - 1), int(i))

            board[2][np.uint8(((dom[0] * (dom[0] + 1)) / 2) + dom[0])] = 1
        return board

    def step(self, action):
        print(action)
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        identities = [(state, actionValues)]

        currentBoard = state.board
        currentAV = actionValues

        """currentBoard = np.array([
            currentBoard[2]
        ])

        currentBoard = np.array([
            currentBoard[2]
        ])"""

        identities.append((GameState(currentBoard, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        


        self.playerTurn = playerTurn
        self.drawCount = 0
        self.player_hand, self.opponent_hand = self._getHands()
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        """print('*****************')
        print("Current hand:  {0}".format(self.player_hand))
        print("Current board: {0}".format(self.board[2]))
        print("Actions: {0}".format(self.allowedActions))
        print('*****************')"""

        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()
        self.passed = False
        self.extra_turn = False

    def _getHands(self):
        if self.playerTurn == 1:
            currentplayer_position = np.zeros(len(self.board[0]), dtype=np.int)
            currentplayer_position[self.board[0] == 1] = 1

            other_position = np.zeros(len(self.board[0]), dtype=np.int)
            other_position[self.board[1] == 1] = 1
        else:
            currentplayer_position = np.zeros(len(self.board[0]), dtype=np.int)
            currentplayer_position[self.board[1] == 1] = 1

            other_position = np.zeros(len(self.board[0]), dtype=np.int)
            other_position[self.board[0] == 1] = 1

        return currentplayer_position, other_position

    def _to_domino(self, index):
        index += 1

        i = .5 * (np.sqrt((8 * (index)) + 1) - 1)

        if i % 1 != 0:
            i = np.floor(i)
            k = index - ((i *(i + 1)) / 2)

            dom = (int(k - 1), int(i))
            return dom
        else:
            dom = (int(i - 1),int(i - 1))
            return dom

    def _head_value(self, index):
        if index == 0:
            return 0
        elif index == 2:
            return 1
        elif index == 5:
            return 2
        elif index == 9:
            return 3
        elif index == 14:
            return 4
        elif index == 20:
            return 5
        elif index == 27:
            return 6
        else:
            print("Failed head_value index: {0}".format(index))

    def _head_index(self, value):
        if value == 0:
            return 0
        elif value == 1:
            return 2
        elif value == 2:
            return 5
        elif value == 3:
            return 9
        elif value == 4:
            return 14
        elif value == 5:
            return 20
        elif value == 6:
            return 27
        else:
            print("Failed head_index value: {0}".format(value))

    def _draw(self):
        bones = np.nonzero(self.board[3])[0]

        if len(bones) > 0:
            self.drawCount += 1

            draw = np.random.choice(bones)

            self.player_hand[draw] = 1
            self.board[3][draw] = 0
            if self.playerTurn == 1:
                self.board[0][draw] = 1
            else:
                self.board[1][draw] = 1

    def _allowedActions(self):
        if np.count_nonzero(self.board[2]) == 0:
            print(self.board)

        head_domino_index = np.nonzero(self.board[2])[0][0]
        head_domino = self._head_value(head_domino_index)

        if np.count_nonzero(self.player_hand) == 0:
            return []

        """if head_domino == 0:
            return np.nonzero(self.player_hand)[0]"""

        while 1:
            hand = np.nonzero(self.player_hand)[0]

            actions = []

            for index in hand:
                domino = self._to_domino(np.int(index))
                if head_domino in domino or 0 in domino or head_domino == 0:
                    actions.append(int(index))

            if np.count_nonzero(self.board[3]) == 0 and len(actions) == 0:
                return []
            elif len(actions) == 0:
                self._draw()
            else:
                return actions

    def _match(self,value,domino):
        if value in domino:
            return True
        else:
            return False

    def _binary(self):
        """if self.playerTurn == 1:
            currentplayer_position = np.zeros(len(self.board[0]), dtype=np.int)
            currentplayer_position[self.board[0] == 1] = 1

            other_position = np.zeros(len(self.board[0]), dtype=np.int)
            other_position[self.board[1] == 1] = 1
        else:
            currentplayer_position = np.zeros(len(self.board[0]), dtype=np.int)
            currentplayer_position[self.board[1] == 1] = 1

            other_position = np.zeros(len(self.board[0]), dtype=np.int)
            other_position[self.board[0] == 1] = 1"""

        """if self.playerTurn == 1:
            currentplayer_position = len(np.nonzero(self.board[0]))

            other_position = len(np.nonzero(self.board[1]))
        else:
            currentplayer_position = len(np.nonzero(self.board[1]))

            other_position = len(np.nonzero(self.board[0]))"""

        position = np.zeros((4, 28), dtype=np.int)
        #print(self.board)

        if self.playerTurn == 1:
            position[0][self.board[0] == 1] = 1
            position[1][self.board[1] == 1] = 1

        else:
            position[0][self.board[1] == 1] = 1
            position[1][self.board[0] == 1] = 1

        #position = np.append(currentplayer_position, other_position)


        position[2] = np.copy(self.board[2])
        position[3] = np.copy(self.board[3])

        return (self.board)

    def _convertStateToId(self):
        #position = np.append(self.player_hand, self.opponent_hand)
        position = np.append(np.count_nonzero(self.player_hand),np.count_nonzero(self.opponent_hand))
        id = ''.join(map(str, position))

        return id

    def _checkForEndGame(self):
        if np.count_nonzero(self.board[0]) == 0 or np.count_nonzero(self.board[1]) == 0:
            return 1
        if len(self.allowedActions) == 0:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if np.count_nonzero(self.opponent_hand) == 0:
            return(-1,-1, 1)

        return (0, 0, 0)

    def _getScore(self):
        count_a = np.count_nonzero(self.player_hand)
        count_a = 28 - count_a

        count_b = np.count_nonzero(self.opponent_hand)
        count_b = 28 - count_b

        return (count_a, count_b)

    def _convert_head(self, action, newBoard):
        played_domino = self._to_domino(np.int(action))

        head_domino_index = np.nonzero(self.board[2])[0]
        head_domino = self._head_value(head_domino_index)

        if played_domino[0] == head_domino:
            newBoard[2][self._head_index(played_domino[1])] = 1
        elif played_domino[1] == head_domino:
            newBoard[2][self._head_index(played_domino[0])] = 1
        elif played_domino[0] == 0:
            newBoard[2][self._head_index(played_domino[1])] = 1
        else:
            newBoard[2][self._head_index(played_domino[0])] = 1

        newBoard[0][action] = 0
        newBoard[1][action] = 0

        return newBoard

    def takeAction(self, action):
        #newBoard = np.copy(self.board)
        newBoard = np.zeros(  # 0 - player hand, 1 - opponent hand, 2 - board, 3-bonepile
            (4, 28), dtype=np.int)
        newBoard[0][self.board[0] == 1] = 1
        newBoard[1][self.board[1] == 1] = 1
        newBoard[3][self.board[3] == 1] = 1
        newBoard = self._convert_head(action, newBoard)

        #if self.extra_turn:
            #newState = GameState(newBoard, self.playerTurn)
        #else:
        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self, logger):
        logger.info("Current Turn: {0}".format(self.playerTurn))
        #print("Current Turn: {0}".format(self.playerTurn))
        logger.info("# of draws this turn: {0}".format(self.drawCount))
        #print("# of draws this turn: {0}".format(self.drawCount))

        doms = []

        hand_indices = np.nonzero(self.board[0])[0]
        for i in hand_indices:
            doms.append(self._to_domino(np.int(i)))

        if len(doms) > 0:
            logger.info("Player 1 hand: {0}".format(doms))   #p1 hand
            #print("Player 1 hand: {0}".format(doms))
        else:
            logger.info("Player 1 hand: Empty")
            #print("Player 1 hand: Empty")

        doms.clear()

        hand_indices = np.nonzero(self.board[1])[0]
        for i in hand_indices:
            doms.append(self._to_domino(np.int(i)))

        if len(doms) > 0:
            logger.info("Player 2 hand: {0}".format(doms))   #p1 hand
            #print("Player 2 hand: {0}".format(doms))
        else:
            logger.info("Player 2 hand: Empty")
            #print("Player 2 hand: Empty")

        board_indices = np.nonzero(self.board[2])[0]

        if len(board_indices) == 0:
            head_domino = -1
        else:
            head_domino = self._head_value(board_indices[0])

        logger.info("Head Domino Value: {0}".format(head_domino))
        #print("Head Domino Value: {0}".format(head_domino))
        doms.clear()

        for i in self.allowedActions:
            doms.append(self._to_domino(i))

        logger.info("Available actions: {0}".format(doms))
        #print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        logger.info('--------------')
        #print('--------------')