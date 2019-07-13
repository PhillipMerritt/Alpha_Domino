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

        hands, collections, passed, marks, tricks_won = self._generate_board()  # generate a new board

        self.gameState = GameState(self.currentPlayer, hands, collections, -1, -1, passed, marks, tricks_won, -1, -1, 0)  # create a GameState
        # action space:
        # 20 for bids 30-42, 84, 168, 336, 672, 1344, 2688, pass
        # 28 for each domino
        # 9 for choosing trump suit (suits 0-6, doubles, follow me)
        self.actionSpace = [np.zeros((20), dtype=np.int), np.zeros((28), dtype=np.int), np.zeros((9),dtype=np.int)]
        self.grid_shape = (16, 28)  # I believe this was just used for printing the conect 4
        self.input_shape = (16, 28)  # input shape for the neural network is ____
        self.name = 'Texas_42'
        self.state_size = len(self.gameState.binary)  # size of the entire game state
        self.action_size = [len(self.actionSpace[0]),len(self.actionSpace[1]),len(self.actionSpace[2])]  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        hands, collections, passed, marks, tricks_won = self._generate_board()

        self.gameState = GameState(self.currentPlayer, hands, collections, -1, -1, passed, marks, tricks_won, -1, -1, 0)

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
        passed = [False,False,False,False]
        marks = [0, 0]
        tricks_won = [0, 0]

        return hands, collections, passed, marks, tricks_won

    # once an action has been chosen this function is called and it keeps making actions until their is a choice to be made
    def step(self, action, logger):  # game state makes the chosen action and returns the next state, the value of the next state for the active player in that state, and whether or not the game is over
        while 1:
            next_state, value, done = self.gameState.takeAction(
                action)  # value is always 0 unless the game is over. Otherwise it is -1 since the last player made a winning move
            self.gameState = next_state  # updates the gameState
            self.currentPlayer = next_state.playerTurn  # swaps current player

            info = None  # idk what this is

            self.gameState.render(logger,action)  # I moved rendering to here so that the automated turns would still be logged
            if done or len(self.gameState.allowedActions) > 1:  # if the game is over or the current player has a choice break the loop
                break
            elif len(self.gameState.allowedActions) == 1:  # else takeAction() with the one action available
                action = self.gameState.allowedActions[0]
            else:  # or if no actions are available pass turn by taking action -1
                print("No available actions!")

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
    def __init__(self, playerTurn, hands, collections, high_bid, highest_bidder, passed,
                 marks, tricks_won, trump_suit, f_m_suit, decision_type, played_dominoes=[-1,-1,-1,-1]):
        # all_domino is a list of tuples containing the pip value for each domino
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.doubles = [0, 2, 5, 9, 14, 20, 27]
        self.honors = {8:5,11:5,15:5,20:10,25:10}
        self.doms_in_suits = {0:[0],1:[1,2],2:[3,4,5],3:[6,7,8,9],4:[10,11,12,13,14],5:[15,16,17,18,19,20],6:[21,22,23,24,25,26,27,28],7:[0, 2, 5, 9, 14, 20, 27]}

        self.possible_bids = list(range(30,43))
        self.possible_bids.extend([84,168,336,672,1344,2688])
        self.bids_to_marks = {84:2,168:3,336:4,672:5,1344:6,2688:7} # dict to look up how many marks are won by winning a trick

        self.hands = hands
        self.played_dominoes = played_dominoes
        self.collections = collections
        self.tricks_won = tricks_won
        self.marks = marks

        self.trump_suit = trump_suit    # 0, 1, 2, 3, 4, 5, 6, 7, 8 (7 = doubles) (8 = follow me)
        self.f_m_suit = f_m_suit

        self.passed = passed
        self.high_bid = high_bid
        self.highest_bidder = highest_bidder

        self.decision_type = decision_type

        self.playerTurn = playerTurn

        self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now


        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move
        #self.score = self._getScore()

    def _allowedActions(self):
        # The player holding the 0-1 tile bids first. Each player may either bid or pass. Each bid must be a number from 30 to 42, and must raise the previous bid. If the bid is maxed out (42), it may be doubled (84), and then doubled again (168). Bids of 42 or greater are made only my taking all 42 points.
        #
        # If no one bids, the tiles are reshuffled and dealt again.
        if self.decision_type == 0: # bidding phase
            actions = [19]  # 19 will represent passing instead of raising the bid and will always be available

            # all bids represented as actions will be 0-16 so they will have 30 subtracted for the non doubled bids
            if not self.passed[self.playerTurn]: # if this player hasn't passed
                if self.high_bid == -1:   # if no one has bid yet this player can bid anywhere from 30-84 and pass
                    for i in range(0, 13):
                        actions.append(i)
                    actions.append(13)
                elif self.high_bid < 42:   # if the bid is below 42 they can be anywhere from the highest bid + 1 to 42
                    for i in range(self.high_bid + 1, 43):
                        actions.append(i - 30)
                    actions.append(13)
                elif self.high_bid == 84: # if the highest bid is 42 they can bid 84
                    actions.append(14)
                elif self.high_bid == 168:   # if it is 84 they can bid 164
                    actions.append(15)
                elif self.high_bid == 336:
                    actions.append(16)
                elif self.high_bid == 672:
                    actions.append(17)
                elif self.high_bid == 1344:
                    actions.append(18)

        elif self.decision_type == 1: # a domino is to be played
            actions = []
            
            if self.f_m_suit == -1: # this player is setting the temp suit
                actions = deepcopy(self.hands[self.playerTurn])
            else:
                for dom in self.hands[self.playerTurn]:
                    pip_tuple = self.all_domino[dom]
                    if self.f_m_suit in pip_tuple:
                        actions.append(dom)

                if len(actions) == 0:  # no dominoes in hand matching follow suit so any domino can be played
                    actions = deepcopy(self.hands[self.playerTurn])
        else:   # a pip or doubles is to be chosen as the trump suit alternatively the player can pass
            actions = list(range(0,9)) # suits 0-6, doubles, follow me

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

        return GameState(self.playerTurn, new_hands, self.collections, self.high_bid, self.highest_bidder, self.passed, self.marks,
                         self.tricks_won, self.trump_suit, self.f_m_suit, self.decision_type, self.played_dominoes)

    def _binary(self):  # converts the state to a 16x28 binary representation
                        # turn sequence is current player, their teammate, then their two opponents
    # hands in turn sequence, collections in turn sequence, bids in turn sequence,board, played domino (for trump suit choosing), trump suit
        position = np.zeros((16, 28), dtype=np.int)

        turn_sequence = [self.playerTurn, (self.playerTurn + 2) % 4,(self.playerTurn + 1) % 4, (self.playerTurn + 3) % 4]

        if self.playerTurn == 0 or self.playerTurn == 2:
            position[0][self.marks[0]] = 1
            position[1][self.marks[1]] = 1
        else:
            position[0][self.marks[1]] = 1
            position[1][self.marks[0]] = 1

        for i, turn in enumerate(turn_sequence):
            for dom in self.hands[turn]:
                position[i+2][dom] = 1


        for dom in self.collections[self.playerTurn % 2]:
            position[6][dom] = 1
        for dom in self.collections[(self.playerTurn + 1) % 2]:
            position[7][dom] = 1

        if self.highest_bidder != -1:
            position[8][self.highest_bidder] = 1

        bin_bid = np.binary_repr(self.high_bid, width=13)    # creates a string of the 13 bit (to allow for 2688) binary representation of the current high bid
        for i,c in enumerate(bin_bid):
            position[9][i] = int(c)

        for i,turn in enumerate(turn_sequence):
            if self.passed[turn]:
                position[10][i] = 1   # set the element to 1 if this player has passed

        for i,dom in enumerate(self.played_dominoes):
            if dom != -1:
                position[i+11][dom] = 1

        if self.trump_suit != -1:
            if self.trump_suit == 7 or self.trump_suit == 8:
                position[15][self.trump_suit] = 1
            else:
                position[15][self.doubles[self.trump_suit]] = 1

        return (position)

    # Creates a string id for the state that looks like:
    def _convertStateToId(self):
        turn_sequence = [self.playerTurn, (self.playerTurn + 2) % 4, (self.playerTurn + 1) % 4,
                         (self.playerTurn + 3) % 4]

        id = str(self.decision_type)

        id += '|' + ''.join(map(str, self.allowedActions))


        if self.playerTurn == 0 or self.playerTurn == 2:
            id += '|' + str(self.marks[0])
            id += '|' + str(self.marks[1])
        else:
            id += '|' + str(self.marks[1])
            id += '|' + str(self.marks[0])

        id += '|' + str(self.high_bid)
        id += '|' + str(self.highest_bidder)


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

        if self.decision_type == 0:
            for tf in self.passed:
                if tf:
                    id += '|T'
                else:
                    id += '|F'
        else:
            for turn in turn_sequence:  # played domino for this trick for each player
                id += '|' + str(self.played_dominoes[turn])


        return id

    def _checkForEndGame(self):  # returns 1 if the last player played their last domino or if the current player has no possible plays otherwise returns 0
        if max(self.marks) >= 7:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state
        if self.marks[0] >= 7:
            return [1, -1]
        elif self.marks[1] >= 7:
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

    # compares two dominoes and returns true if dom_b is heavier than domino
    def compare_doms(self, dom_a, dom_b):
        if self.trump_suit == 7 and dom_a[0] == dom_a[1]: # if the trump suit is doubles and this dom is a double
            dom_weight_a = 17   # highest possible weight
        elif self.trump_suit in dom_a:  # if the dom is in the trump suit
            if dom_a[0] == self.trump_suit: # make the  opposite pip from the trump suit pip the domino's weight + 10
                dom_weight_a = dom_a[1] + 10
            else:
                dom_weight_a = dom_a[0] + 10
        else:   # if it isn't in the trump suit store it's weight
            if self.f_m_suit in dom_a and dom_a[0] == dom_a[0]:
                dom_weight_a = 7    # highest possible non-trump weight
            elif dom_a[0] == self.f_m_suit:
                dom_weight_a = dom_a[1]
            elif dom_a[1] == self.f_m_suit:
                dom_weight_a = dom_a[0]
            else:   # if the domino isn't in either suit then it's value is 0
                dom_weight_a = 0

        if self.trump_suit == 7 and dom_b[0] == dom_b[1]:  # if the trump suit is doubles and this dom is a double
            dom_weight_b = 17  # highest possible weight
        elif self.trump_suit in dom_b:  # if the dom is in the trump suit
            if dom_b[0] == self.trump_suit:  # make the  opposite pip from the trump suit pip the domino's weight + 10
                dom_weight_b = dom_b[1] + 10
            else:
                dom_weight_b = dom_b[0] + 10
        else:  # if it isn't in the trump suit store it's weight
            if self.f_m_suit in dom_b and dom_b[0] == dom_b[0]:
                dom_weight_b = 7  # highest possible non-trump weight
            elif dom_b[0] == self.f_m_suit:
                dom_weight_b = dom_b[1]
            elif dom_b[1] == self.f_m_suit:
                dom_weight_b = dom_b[0]
            else:  # if the domino isn't in either suit then it's value is 0
                dom_weight_b = 0

        return dom_weight_a < dom_weight_b

    # finds out the winner of the trick and adds the honors of the trick to that team's collection
    def trick_score(self, played_dominoes):
        heaviest = self.all_domino[self.played_dominoes[0]]              # tracks heaviest domino played
        winning_player = 0
        new_collections = deepcopy(self.collections)    # copy of collections to add honors too

        for i in range(1,4):
            if self.compare_doms(heaviest,self.all_domino[self.played_dominoes[i]]):
                heaviest = self.all_domino[self.played_dominoes[i]]
                winning_player = i

        winning_team = winning_player % 2

        for dom in played_dominoes: # honors played this trick go to the winner of the trick
            if dom in self.honors:
                new_collections[winning_team].append(dom)

        new_tricks_won = deepcopy(self.tricks_won)
        new_tricks_won[winning_team] += 1

        return new_tricks_won, new_collections, winning_player

    def score_collections(self,collections):
        points = [0,0]
        for i,col in enumerate(collections):
            for dom in col:
                points[i]+=self.honors[dom]

        return points

    def takeAction(self, action):
        if action not in self.allowedActions:
            print("Illegal Action!")

        new_hands = [[], [], [], []]

        if self.decision_type == 0: # bid action
            new_hands = deepcopy(self.hands)
            new_high_bid = self.high_bid
            new_passed = deepcopy(self.passed)
            next_player = (self.playerTurn + 1) % 4
            highest_bidder = self.highest_bidder

            if action == 18:
                new_high_bid = self.possible_bids[action]
                highest_bidder = self.playerTurn
                next_player = highest_bidder
                new_passed = [True,True,True,True]
            elif action != 19:
                new_high_bid = self.possible_bids[action]
                highest_bidder = self.playerTurn
            else:
                new_passed[self.playerTurn] = True

            if False not in new_passed:
                scrap_hand = True
                if self.high_bid > 0:
                    scrap_hand = False

                if scrap_hand:
                    next_d_type = 0
                    new_hands = self.deal_hands()
                    new_passed = [False,False,False,False]
                else:
                    next_d_type = 2
                    next_player = highest_bidder
                    highest_bidder = highest_bidder % 2 # changed to team index rather than player after bidding phase
            else:
                next_d_type = 0
                for i in range(1,5):
                    if not new_passed[(self.playerTurn + i) % 4]:
                        next_player = (self.playerTurn + i) % 4
                        break

            newState = GameState(next_player, new_hands, self.collections, new_high_bid, highest_bidder, new_passed, self.marks, self.tricks_won, self.trump_suit, self.f_m_suit,
                         next_d_type)  # create new state
        elif self.decision_type == 1:   # play domino action
            new_hands = deepcopy(self.hands)
            new_collections = deepcopy(self.collections)

            new_hands[self.playerTurn].remove(action)

            new_played = deepcopy(self.played_dominoes)
            new_played[self.playerTurn] = action

            if self.f_m_suit == -1:
                new_f_m = max(self.all_domino[action])
            else:
                new_f_m = self.f_m_suit


            if -1 not in new_played:    # trick is over
                new_tricks_won, new_collections, winner = self.trick_score(new_played)
                added_points = self.score_collections(new_collections)
                added_points[0] += new_tricks_won[0]
                added_points[1] += new_tricks_won[1]
                new_marks = deepcopy(self.marks)
                hand_complete = False

                if self.high_bid >= 42:
                    points_to_win = 42
                    points_to_block = 1
                else:
                    points_to_win = self.high_bid
                    points_to_block = 42 - self.high_bid + 1

                if (self.highest_bidder == 0 and added_points[0] >= points_to_win) or (self.highest_bidder == 1 and added_points[0] >= points_to_block):
                    hand_complete = True


                    if self.high_bid <= 42:
                        new_marks[0] += 1
                    else:
                        new_marks[0] += self.bids_to_marks[self.high_bid]
                elif (self.highest_bidder == 1 and added_points[1] >= points_to_win) or (self.highest_bidder == 0 and added_points[1] >= points_to_block):
                    hand_complete = True

                    if self.high_bid <= 42:
                        new_marks[1] += 1
                    else:
                        new_marks[1] += self.bids_to_marks[self.high_bid]

                if hand_complete: # if the trick is over and anyone has an empty hand this hand is over
                    new_passed = [False, False, False, False]
                    new_collections = [[],[]]
                    new_tricks_won = [0,0]
                    new_hands = self.deal_hands()
                    newState = GameState(winner, new_hands, new_collections, -1, -1, new_passed, new_marks, new_tricks_won, -1, -1, 0)
                else:
                    newState = GameState(winner, new_hands, new_collections, self.high_bid, self.highest_bidder,
                                         self.passed, self.marks, new_tricks_won, self.trump_suit, -1, 1)
            else:
                newState = GameState((self.playerTurn+1)%4, new_hands, new_collections, self.high_bid, self.highest_bidder,
                                     self.passed, self.marks, self.tricks_won, self.trump_suit, new_f_m, 1, new_played)
        else:
            new_suit = action

            newState = GameState(self.playerTurn, self.hands, self.collections, self.high_bid, self.highest_bidder,
                                 self.passed, self.marks, self.tricks_won, new_suit, -1, 1)
        value = newState.value
        done = 0

        if newState.isEndGame:  # if the game is over in the new state store its value to value and update done to 1
            done = 1


        return (newState, value, done)

    def render(self, logger, action = None):  # this logs each gamestate to a logfile in the run folder. The commented sections will print the game states to the terminal if you uncomment them
        turn_sequence = [0, 2, 1, 3]

        logger.info('--------------')
        logger.info("Current Turn: {0} | DECISION TYPE: {1}".format(self.playerTurn, self.decision_type))
        if self.decision_type == 0:
            logger.info("Available Actions: {0}".format(self.allowedActions))

        elif self.decision_type == 1:
            temp_hand = []

            for index in self.allowedActions:
                temp_hand.append((index, self.all_domino[index]))

            logger.info("Available Actions: {0}".format(temp_hand))
        else:
            logger.info("Available Actions: {0}".format(self.allowedActions))

        logger.info("Current Marks: Team 1 = {0}/7, Team 2 = {1}/7".format(self.marks[0],self.marks[1]))
        logger.info("Tricks Won: Team 1 = {0}/7, Team 2 = {1}/7".format(self.tricks_won[0],self.tricks_won[1]))

        # logger.info("{0}".format())
        temp_hand = []
        temp_hand_2 = []

        for index in self.collections[0]:
            temp_hand.append(index)
        for index in self.collections[1]:
            temp_hand_2.append(index)

        logger.info("Team 1 Collection: {0} | Team 2 Collection: {1}".format(temp_hand,temp_hand_2))

        if self.decision_type == 0:
            logger.info("Passed: {0}".format(self.passed))

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

        if self.high_bid > 0:
            logger.info("Highest Bidder: {0} Highest Bid: {1}".format(self.highest_bidder,self.high_bid))

        if self.decision_type != 0:
            logger.info("Trump Suit: {0}".format(self.trump_suit))
            logger.info("FM Suit: {0}".format(self.f_m_suit))

            for i, turn in enumerate(turn_sequence):
                if i == 0:
                    logger.info("Team 1 Played Dominoes:")
                elif i == 2:
                    logger.info("Team 2 Played Dominoes:")

                logger.info("Player {0}: {1}".format(turn, self.all_domino[self.played_dominoes[turn]]))

        logger.info('--------------')

    def user_print(self):
        turn_sequence = [0, 2, 1, 3]

        print('--------------')
        print("Current Turn: {0} | DECISION TYPE: {1}".format(self.playerTurn, self.decision_type))
        print("Current Marks: Team 1 = {0}/7, Team 2 = {1}/7".format(self.marks[0], self.marks[1]))
        print("Tricks Won: Team 1 = {0}/7, Team 2 = {1}/7".format(self.tricks_won[0], self.tricks_won[1]))

        # print("{0}".format())
        temp_hand = []
        temp_hand_2 = []

        for index in self.collections[0]:
            temp_hand.append(index)
        for index in self.collections[1]:
            temp_hand_2.append(index)

        print("Team 1 Collection: {0} | Team 2 Collection: {1}".format(temp_hand, temp_hand_2))

        if self.decision_type == 0:
            print("Passed: {0}".format(self.passed))

        for i, turn in enumerate(turn_sequence):
            temp_hand = []
            if i == 0:
                print("Team 1 Hands:")
            elif i == 2:
                print("Team 2 Hands:")

            for index in self.hands[turn]:
                temp_hand.append(self.all_domino[index])

            if turn == self.playerTurn:
                print("Player {0}: {1} <<<< Active Player".format(turn, temp_hand))
            else:
                print("Player {0}: {1}".format(turn, temp_hand))

        if self.high_bid > 0:
            print("Highest Bidder: {0} Highest Bid: {1}".format(self.highest_bidder, self.high_bid))

        if self.decision_type != 0:
            print("Trump Suit: {0}".format(self.trump_suit))
            print("FM Suit: {0}".format(self.f_m_suit))


            for i, turn in enumerate(turn_sequence):
                if i == 0:
                    print("Team 1 Played Dominoes:")
                elif i == 2:
                    print("Team 2 Played Dominoes:")

                print("Player {0}: {1}".format(turn, self.all_domino[self.played_dominoes[turn]]))
        if self.decision_type == 0:
            print("Available Actions: {0}".format(self.allowedActions))
        elif self.decision_type == 1:
            temp_hand = []

            for index in self.allowedActions:
                temp_hand.append((index, self.all_domino[index]))

            print("Available Actions: {0}".format(temp_hand))
        else:
            print("Available Actions: {0}".format(self.allowedActions))
        print('--------------')