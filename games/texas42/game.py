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

        hands, collections, bids, passed, points = self._generate_board()  # generate a new board

        self.gameState = GameState(self.currentPlayer, hands, collections, bids, passed, points, -1, 0)  # create a GameState
        # action space:
        # 16 for bids 30-42, 84, 168, pass
        # 28 for each domino
        # 4 for choosing trump suit (0:pip1, 1:pip2, 2:doubles, 3:pass)
        self.actionSpace = [np.zeros((16), dtype=np.int), np.zeros((28), dtype=np.int), np.zeros((4),dtype=np.int)]
        self.grid_shape = (15, 28)  # grid shape is 1x28
        self.input_shape = (15, 28)  # input shape for the neural network is 5x28
        self.name = 'Texas_42'
        self.state_size = len(self.gameState.binary)  # size of the entire game state
        self.action_size = [len(self.actionSpace[0]),len(self.actionSpace[1]),len(self.actionSpace[2])]  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        hands, collections, bids, passed, points = self._generate_board()

        self.gameState = GameState(self.currentPlayer, hands, collections, bids, passed, points, -1, 0)

        return self.gameState

    def _generate_board(self):
        hands = [[], [], [], []]
        queue = globals.queue_reset()  # reset and shuffle the queue

        for i in range(7):
            hands[0].append(queue.pop())  # pop 7 doms off the queue for each player's hand
            hands[1].append(queue.pop())
            hands[2].append(queue.pop())
            hands[3].append(queue.pop())

        for i, hand in enumerate(hands):  # whoever has the 0:1 domino bids first
            hands[i] = sorted(hand)
            if hands[i][0] == 1 or hands[i][1] == 1:
                self.currentPlayer = i


        collections = [[],[]]
        bids = [0, 0, 0, 0]
        passed = [False,False,False,False]
        points = [0, 0]

        return hands, collections, bids, passed, points

    # once an action has been chosen this function is called and it keeps making actions until their is a choice to be made
    def step(self, action, logger):  # game state makes the chosen action and returns the next state, the value of the next state for the active player in that state, and whether or not the game is over
        while 1:
            next_state, value, done = self.gameState.takeAction(
                action)  # value is always 0 unless the game is over. Otherwise it is -1 since the last player made a winning move
            self.gameState = next_state  # updates the gameState
            self.currentPlayer = next_state.playerTurn  # swaps current player

            info = None  # idk what this is

            self.gameState.render(logger,action)  # I moved rendering to here so that the automated turns would still be logged
            if done or len(
                    self.gameState.allowedActions) > 1:  # if the game is over or the current player has a choice break the loop
                break
            elif len(self.gameState.allowedActions) == 1:  # else takeAction() with the one action available
                action = self.gameState.allowedActions[0]
            else:  # or if no actions are available pass turn by taking action -1
                action = -1

        return ((next_state, value, done, info))

    def identities(self, state, actionValues):  # haven't looked into what this function is doing quite yet
        identities = [(state, actionValues)]

        """ currentHands = state.hands
        currentBids = state.bids
        currentPlayed = state.played_dominoes
        currentAV = actionValues

        identities.append(
            (GameState(currentHands, currentPlayed, currentBids, state.playerTurn), currentAV))
        """
        return identities


class GameState():
    def __init__(self, playerTurn, hands, collections, bids, passed, points, trump_suit, decision_type, played_dominoes=[-1,-1,-1,-1]):
        # all_domino is a list of tuples containing the pip value for each domino
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.doubles = [0, 2, 5, 9, 14, 20, 27]
        self.honors = {8:5,11:5,15:5,20:10,25:10}

        self.possible_bids = list(range(30,43))
        self.possible_bids.extend([84,168])

        self.hands = hands
        self.played_dominoes = played_dominoes
        self.collections = collections
        self.points = points

        self.trump_suit = trump_suit    # 0, 1, 2, 3, 4, 5, 6, 7 (7 = doubles)

        self.bids = bids
        self.passed = passed
        self.highest_bid = max(bids)
        self.highest_bidder = np.argmax(bids)

        self.decision_type = decision_type

        self.playerTurn = playerTurn
        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now

        self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move
        #self.score = self._getScore()

    def _allowedActions(self):
        # The player holding the 0-1 tile bids first. Each player may either bid or pass. Each bid must be a number from 30 to 42, and must raise the previous bid. If the bid is maxed out (42), it may be doubled (84), and then doubled again (168). Bids of 42 or greater are made only my taking all 42 points.
        #
        # If no one bids, the tiles are reshuffled and dealt again.
        if self.decision_type == 0: # bidding phase
            actions = [15]  # 15 will represent passing instead of raising the bid and will always be available

            # all bids represented as actions will be 0-16 so they will have 30 subtracted for the non doubled bids
            if not self.passed[self.playerTurn]: # if this player hasn't passed
                if self.highest_bid == 0:   # if no one has bid yet this player can bid anywhere from 30-42
                    actions = []
                    for i in range(0, 13):
                        actions.append(i)
                elif self.highest_bid < 42:   # if the bid is below 42 they can be anywhere from the highest bid + 1 to 42
                    for i in range(self.highest_bid + 1, 43):
                        actions.append(i - 30)
                elif self.highest_bid == 42: # if the highest bid is 42 they can bid 84
                    actions.append(13)
                elif self.highest_bid == 84:   # if it is 84 they can bid 164
                    actions.append(14)
        elif self.decision_type == 1: # a domino is to be played
            actions = []
            heavy_suits = []

            if self.trump_suit == -1: # a trump suit hasn't been chosen
                actions = deepcopy(self.hands[self.playerTurn])
            elif self.trump_suit == 7:  # doubles
                for dom in self.hands[self.playerTurn]:
                    if dom in self.doubles: # add any doubles in hand to available actions
                        actions.append(dom)

                if len(actions) == 0:   # if no doubles in hand then the player can play any domino
                    actions = deepcopy(self.hands[self.playerTurn])

            else:   # all dominoes in hand that match the trump suit
                for dom in self.hands[self.playerTurn]:
                    pip_tuple = self.all_domino[dom]
                    if self.trump_suit in pip_tuple:
                        actions.append(dom)

                if len(actions) == 0:   # no dominoes in hand matching trump suit so any domino can be played
                    actions = deepcopy(self.hands[self.playerTurn])
        else:   # a pip or doubles is to be chosen as the trump suit alternatively the player can pass
            actions = []   # 3 is passing

            pip_tuple = self.all_domino[self.played_dominoes[self.playerTurn]]

            if pip_tuple[0] != pip_tuple[1]:    # if it isn't a double
                actions.append(0)   # either pip
                actions.append(1)
            else:
                actions.append(0)   # the double value
                actions.append(2)   # double suit

            #actions.append(3)

        actions = sorted(actions)

        return actions

    # creates a list of hidden information by adding the opponent's hand back into the queue
    # then generate a cloned gameState with the opponents hand generated from the shuffled
    # unknown list
    def CloneAndRandomize(self):
        unknown = []
        for i in range(0, 4):
            if i != self.playerTurn:
                unknown.extend(self.hands[i])

        new_hands = [[], [], [], []]

        for dom in self.hands[self.playerTurn]:  # copy over the current players hand
            new_hands[self.playerTurn].append(dom)

        np.random.shuffle(unknown)

        for i in range(0, 4):
            if i != self.playerTurn:
                for j in range(len(self.hands[i])):
                    new_hands[i].append(unknown.pop())

        for i,hand in enumerate(new_hands):
            if i != self.playerTurn:
                new_hands[i] = sorted(hand)

        return GameState(self.playerTurn, new_hands, self.collections, self.bids, self.passed, self.points, self.trump_suit,
                         self.decision_type, self.played_dominoes)

    def _binary(self):  # converts the state to a 6x28 binary representation
                        # turn sequence is current player, their teammate, then their two opponents
    # hands in turn sequence, collections in turn sequence, bids in turn sequence,board, played domino (for trump suit choosing), trump suit
        position = np.zeros((15, 28), dtype=np.int)

        turn_sequence = [self.playerTurn, (self.playerTurn + 2) % 4,(self.playerTurn + 1) % 4, (self.playerTurn + 3) % 4]

        for i, turn in enumerate(turn_sequence):
            for dom in self.hands[turn]:
                position[i][dom] = 1


        for dom in self.collections[self.playerTurn % 2]:
            position[4][dom] = 1
        for dom in self.collections[(self.playerTurn + 1) % 2]:
            position[5][dom] = 1

        for i, turn in enumerate(turn_sequence):
            bin_bid = np.binary_repr(self.bids[turn], width=9)    # creates a string of the 9 bit (to allow for 168) binary representation of the current players bid
            for i,c in enumerate(bin_bid):
                position[i+6][i] = int(c)

            if self.passed[turn]:
                position[i+6][27] = 1   # set the last element to 1 if this player has passed

        for i,dom in enumerate(self.played_dominoes):
            if dom != -1:
                position[i+10][dom] = 1

        if self.trump_suit != -1:
            if self.trump_suit == 7:
                position[14][7] = 1
            else:
                position[14][self.doubles[self.trump_suit]] = 1

        return (position)

    # Creates a string id for the state that looks like:
    def _convertStateToId(self):
        turn_sequence = [self.playerTurn, (self.playerTurn + 2) % 4, (self.playerTurn + 1) % 4,
                         (self.playerTurn + 3) % 4]

        id = str(self.decision_type)


        for i in range(0, 2):
            id += '|' + str(self.points[i])

        id += '|' + ''.join(map(str, self.hands[self.playerTurn]))  # create a string version of the hand

        """for i in range(1,4):
            id += '|' + ''.join(map(str, self.hands[(self.playerTurn + i) % 4]))"""

        id += '|' + str(len(self.hands[(self.playerTurn + 2) % 4]))  # add the delimiter and the size of the partner's hand

        id += '|' + str(len(self.hands[(self.playerTurn + 1) % 4]))  # add the delimiter and the size of the opponent's hand

        id += '|' + str(len(self.hands[(self.playerTurn + 3) % 4]))  # add the delimiter and the size of the opponent's hand

        # collections
        if self.collections[self.playerTurn % 2] == []:
            id += '|' + str(-1)
        else:
            id += '|' + ''.join(map(str,self.collections[self.playerTurn % 2]))

        if self.collections[(self.playerTurn + 1) % 2] == []:
            id += '|' + str(-1)
        else:
            id += '|' + ''.join(map(str, self.collections[(self.playerTurn + 1) % 2]))

        id += '|' + str(self.trump_suit)

        for turn in turn_sequence:  # played domino for this trick for each player
            id += '|' + str(self.played_dominoes[turn])


        return id

    def _checkForEndGame(self):  # returns 1 if the last player played their last domino or if the current player has no possible plays otherwise returns 0
        if max(self.points) >= 250:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state
        if self.points[0] >= 250:
            return [1, -1]
        elif self.points[1] >= 250:
            return [-1, 1]
        else:
            return [0,0]

    def deal_hands(self):
        hands = [[], [], [], []]
        queue = globals.queue_reset()  # reset and shuffle the queue

        for i in range(7):
            hands[0].append(queue.pop())  # pop 7 doms off the queue for each player's hand
            hands[1].append(queue.pop())
            hands[2].append(queue.pop())
            hands[3].append(queue.pop())

        for i,hand in enumerate(hands):
            hands[i] = sorted(hand)

        return hands

    def trick_score(self, played_dominoes):
        scored_points = [0,0]
        heaviest = -1
        winning_player = -1
        new_collections = deepcopy(self.collections)

        for i, dom in enumerate(played_dominoes):
            double = False
            pip_tuple = self.all_domino[dom]

            if pip_tuple[0] == pip_tuple[1]:
                double = True

            if pip_tuple[0] == self.trump_suit:
                if double:
                    winning_player = i
                    break
                elif pip_tuple[1] > heaviest:
                    heaviest = pip_tuple[1]
                    winning_player = i
            elif pip_tuple[1] == self.trump_suit:
                if double:
                    winning_player = i
                    break
                elif pip_tuple[0] > heaviest:
                    heaviest = pip_tuple[0]
                    winning_player = i
            elif double and self.trump_suit == 7:
                if dom == 27:
                    winning_player = i
                    break
                elif pip_tuple[0] > heaviest:
                    heaviest = pip_tuple[0]
                    winning_player = i

        if winning_player == -1: # I don't think you can have a trick where no one plays in the trump suit
            print("No one actually played in the trump suit! Fix trick_score in game.py")

        winning_team = winning_player % 2

        for dom in played_dominoes: # honors played this trick go to the winner of the trick
            if dom in self.honors:
                scored_points[winning_team]+=self.honors[dom]
                new_collections[winning_team].append(dom)

        scored_points[winning_team] += 1

        return [self.points[0]+scored_points[0], self.points[1]+scored_points[1]], new_collections, winning_player

    def takeAction(self, action):
        new_hands = [[], [], [], []]

        if self.decision_type == 0: # bid action
            new_hands = deepcopy(self.hands)
            new_bids = deepcopy(self.bids)
            new_passed = deepcopy(self.passed)
            next_player = (self.playerTurn + 1) % 4

            if action != 15:
                new_bids[self.playerTurn] = self.possible_bids[action]
            else:
                new_passed[self.playerTurn] = True

            if False not in new_passed:
                scrap_hand = True
                for bid in self.bids:
                    if bid > 0:
                        scrap_hand = False
                        break

                if scrap_hand:
                    next_d_type = 0
                    new_hands = self.deal_hands()
                    new_passed = [False,False,False,False]
                else:
                    next_d_type = 1
                    next_player = np.argmax(new_bids)
            else:
                next_d_type = 0
                for i in range(1,5):
                    if not new_passed[(self.playerTurn + i) % 4]:
                        next_player = (self.playerTurn + i) % 4
                        break

            newState = GameState(next_player, new_hands, self.collections, new_bids, new_passed, self.points, self.trump_suit,
                         next_d_type)  # create new state
        elif self.decision_type == 1:   # play domino action
            new_hands = deepcopy(self.hands)
            new_collections = deepcopy(self.collections)

            new_hands[self.playerTurn].remove(action)

            new_played = deepcopy(self.played_dominoes)
            new_played[self.playerTurn] = action


            if -1 not in new_played:    # trick is over
                new_points, new_collections, winner = self.trick_score(new_played)

                if len(new_hands[0]) == 0: # if the trick is over and anyone has an empty hand this hand is over
                    new_passed = [False, False, False, False]
                    new_collections = [[],[]]
                    new_hands = self.deal_hands()
                    new_bids = [0,0,0,0]
                    newState = GameState(winner, new_hands, new_collections, new_bids, new_passed, new_points, -1, 0)
                else:
                    newState = GameState(winner, new_hands, new_collections, self.bids, self.passed, new_points, -1, 1)
            elif self.trump_suit == -1: # needs to choose trump suit (got to decision type 3)
                newState = GameState(self.playerTurn, new_hands, new_collections, self.bids, self.passed, self.points, self.trump_suit, 2, new_played)
            else:
                newState = GameState((self.playerTurn+1)%4, new_hands, new_collections, self.bids, self.passed, self.points, self.trump_suit, 1, new_played)
        else:
            if action == 2:
                new_suit = 7
            elif action != 3:
                pip_tuple = self.all_domino[self.played_dominoes[self.playerTurn]]
                new_suit = pip_tuple[action]
            else:
                new_suit = -1

            newState = GameState((self.playerTurn + 1) % 4, self.hands, self.collections, self.bids, self.passed, self.points,
                                 new_suit, 1, self.played_dominoes)

        value = 0
        done = 0

        if newState.isEndGame:  # if the game is over in the new state store its value to value and update done to 1
            value = newState.value
            done = 1

        return (newState, value, done)

    def render(self, logger, action = None):  # this logs each gamestate to a logfile in the run folder. The commented sections will print the game states to the terminal if you uncomment them
        turn_sequence = [0, 2, 1, 3]

        logger.info('--------------')
        logger.info("Current Turn: {0}".format(self.playerTurn))
        if self.decision_type == 0:
            logger.info("Available Actions: {0}".format(self.allowedActions))

        elif self.decision_type == 1:
            temp_hand = []

            for index in self.allowedActions:
                temp_hand.append((index, self.all_domino[index]))

            logger.info("Available Actions: {0}".format(temp_hand))
        else:
            logger.info("Available Actions: {0}".format(self.allowedActions))


        logger.info("Current Scores: Team 1={0}, Team 2={1}".format(self.points[0], self.points[1]))
        if action != None:
            logger.info("Chosen Action: {0}".format(action))

        logger.info("Current Scores: Team 1 = {0}, Team 2 = {1}".format(self.points[0],self.points[1]))
        # logger.info("{0}".format())

        for i in range(0,2):
            temp_hand = []

            for index in self.collections[i]:
                temp_hand.append(index)

            if i == 0:
                logger.info("Team 1 Collection: {0}".format(temp_hand))
            elif i == 1:
                logger.info("Team 2 Collection: {0}".format(temp_hand))



        logger.info("DECISION TYPE: {0}".format(self.decision_type))

        for i,turn in enumerate(turn_sequence):
            temp_hand = []
            if i == 0:
                logger.info("Team 1 Hands:")
            elif i == 2:
                logger.info("Team 2 Hands:")

            for index in self.hands[turn]:
                temp_hand.append(self.all_domino[index])

            if turn == self.playerTurn:
                logger.info("Player {0}: {1} <<<< Active Player".format(turn, temp_hand))
            else:
                logger.info("Player {0}: {1}".format(turn, temp_hand))

        for i, turn in enumerate(turn_sequence):
            if i == 0:
                logger.info("Team 1 Bids:")
            elif i == 2:
                logger.info("Team 2 Bids:")

            logger.info("Player {0}: {1}".format(turn, self.bids[turn]))

        if self.decision_type != 0:
            logger.info("Trump Suit: {0}".format(self.trump_suit))

            for i, turn in enumerate(turn_sequence):
                if i == 0:
                    logger.info("Team 1 Played Dominoes:")
                elif i == 2:
                    logger.info("Team 2 Played Dominoes:")

                logger.info("Player {0}: {1}".format(turn, self.played_dominoes[turn]))

        logger.info('--------------')

    def user_print(self):
        turn_sequence = [0, 2, 1, 3]

        print('--------------')
        print("Current Turn: {0}".format(self.playerTurn))
        if self.decision_type == 0:
            print("Available Actions: {0}".format(self.allowedActions))

        elif self.decision_type == 1:
            temp_hand = []

            for index in self.allowedActions:
                temp_hand.append((index, self.all_domino[index]))

            print("Available Actions: {0}".format(temp_hand))
        else:
            print("Available Actions: {0}".format(self.allowedActions))


        print("Current Scores: Team 1={0}, Team 2={1}".format(self.points[0], self.points[1]))
        # logger.info("{0}".format())

        temp_hand = []

        for index in self.collections[0]:
            temp_hand.append(self.all_domino[index])

        print("Team 1 Collection: {0}".format(temp_hand))

        temp_hand = []

        for index in self.collections[1]:
            temp_hand.append(self.all_domino[index])

        print("Team 2 Collection: {0}".format(temp_hand))

        print("DECISION TYPE: {0}".format(self.decision_type))

        for i, turn in enumerate(turn_sequence):
            temp_hand = []
            if i == 0:
                print("Team 1 Hands:")
            elif i == 2:
                print("Team 2 Hands:")

            for index in self.hands[turn]:
                temp_hand.append(self.all_domino[index])

            if turn == self.playerTurn:
                print("Player {0}: {1} <<<<<<".format(turn, temp_hand))
            else:
                print("Player {0}: {1}".format(turn, temp_hand))

        print("Highest Bidder: {0}".format(np.argmax(self.bids)))

        for i, turn in enumerate(turn_sequence):
            if i == 0:
                print("Team 1 Bids:")
            elif i == 2:
                print("Team 2 Bids:")

            print("Player {0}: {1}".format(turn, self.bids[turn]))

        if self.decision_type != 0:
            print("Trump Suit: {0}".format(self.trump_suit))

            for i, turn in enumerate(turn_sequence):
                if i == 0:
                    print("Team 1 Played Dominoes:")
                elif i == 2:
                    print("Team 2 Played Dominoes:")

                print("Player {0}: {1}".format(turn, self.played_dominoes[turn]))

        print('--------------')