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
        self.actionSpace = [np.zeros(  # action space is 28 * each train
            (28 * len(trains)), dtype=np.int)]
        #self.grid_shape = (4, 28)  # grid shape is 7x28
        #self.input_shape = self.grid_shape = self.gameState.binary.shape  # input shape for the neural network is the shape of the binary state representation
        self.input_shape = self.grid_shape = (2 * PLAYER_COUNT + 3, 28)
        self.name = 'mexican_train'
        self.state_size = len(self.gameState.binary)  # size of the entire game state # TODO: look into this to see if it could be effecting anything

        self.action_size = [len(self.actionSpace[0])]  # size of the actionSpace

    def reset(self):  # creates new game
        count = 1
        while 1:    # sometimes game just play out to completion without any choices being made so games are created until this doesn't happen
            hands, trains, queue = self._generate_board()

            self.gameState = GameState(hands, trains, queue, self.currentPlayer)

            if len(self.gameState.allowedActions) == 0:
                self.step(-1)
            elif len(self.gameState.allowedActions) == 1:
                self.step(self.gameState.allowedActions[0])
            
            if not self.gameState.isEndGame:
                break
            else:
                count += 1

        return self.gameState

    # deal 3 dominoes to each player then choose the starting player based on who has the highest double
    def _generate_board(self):
        highest_double = None

        while not highest_double:
            hands = [[] for players in range(PLAYER_COUNT)]
            queue = globals.queue_reset()   # reset and shuffle the queue

            for i in range(5):
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
    def step(self, action, logger = None):  # game state makes the chosen action and returns the next state, the value of the next state for the active player in that state, and whether or not the game is over
        if type(action) == tuple:
            print(action)
            exit(0)
        while 1:
            next_state, value, done = self.gameState.takeAction(
                action)  # value is always 0 unless the game is over. Otherwise it is -1 since the last player made a winning move
            self.gameState = next_state  # updates the gameState
            self.currentPlayer = self.gameState.playerTurn  # swaps current player
            info = None  # idk what this is
            
            if logger:
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

        return identities


class GameState():
    def __init__(self, hands, trains, queue, playerTurn, passed = [False for player in range(PLAYER_COUNT)]):
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

        self.passed = passed

        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move

        empty_hand = False

        for hand in hands:
            if len(hand) == 0:
                empty_hand = True

        self.playerTurn = playerTurn
        self.drawCount = 0  # tracks the # of times this player has drawn this turn. only used for logging
        self.public_id = self.get_public_info()
        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now

        self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        if len(self.allowedActions) != 0:
            self.passed[self.playerTurn] = False

        if not self.isEndGame and empty_hand:
            print("empty hand yet game not over")

        self.decision_type = 0

    def _draw(self):  # draws a random domino then updates binary and id. If there are no more dominos to draw return false

        if len(self.queue) > 0:  # if there are dominoes to draw
            self.drawCount += 1

            self.hands[self.playerTurn].append(self.queue.pop())  # randomly pop one from the boneyard and place it in the players hand

            self.binary = self._binary()
            self.id = self._convertStateToId()

            return True

        return False

    # generates a list of all allowed actions. If there are no available actions dominoes are drawn if available
    # until there are actions to be made. The actions are in the form of action = (train num * 28) + action (ex. domino 14 to train 3 would be (3*28)+14 = 98
    def _allowedActions(self):  # TODO: every player has to finish a public double if they can or their train becomes public too
        heads = []              # TODO: add wild blanks

        # check to see if any trains have an unfinished double
        # players have to finish doubles if they can unless they
        # played the double the previous turn on a train that isn't theirs
        for i in range(PLAYER_COUNT + 1):
            if i < PLAYER_COUNT:
                index = (self.playerTurn + i) % PLAYER_COUNT
            else:
                index = i

            if self.trains[index].unfinished:
                # if the unfinished train belongs to this player and it isn't marked 
                # then they played the double last turn and must finish it
                if i == 0 and not self.trains[index].marked:
                    heads.append((0, self.trains[index].head))
                    break
                else:
                    heads.append((i, self.trains[index].head))
        
        if heads == []: # if the player isn't forced to play on specific trains
            for i in range(PLAYER_COUNT):  # create a list of the available head values
                index = (self.playerTurn + i) % PLAYER_COUNT
                if i==0 or self.trains[index].marked:
                    heads.append((i, self.trains[index].head))
            
            heads.append((PLAYER_COUNT, self.trains[PLAYER_COUNT].head))    # mexican train

        # check for legal actions. If none found draw a domino and try again. If still none found pass turn and mark this train
        actions = []

        for dom_index in self.hands[self.playerTurn]: # for each domino in hand
            for (i, head) in heads:
                if self.match_check(dom_index, head):
                    actions.append(i * 28 + dom_index)

            

        if len(actions) > 0:  # if there are any available actions return them
            return actions
        elif not self._draw():  # if no actions found draw a domino
            self.passed[self.playerTurn] = True
            return []  # if drawing a domino fails return an empty list

        new_dom = self.hands[self.playerTurn][-1]   # get the drawn domino
        for (i, head) in enumerate(heads):
            if self.match_check(new_dom, head):
                actions.append(i * 28 + new_dom)
        
        if len(actions) > 0:
            return actions
        
        self.trains[self.playerTurn].mark()
        return []

    # function to determine if a domino can be played on the given head value
    def match_check(self, dom, head):
        if head in self.all_domino[dom]:
            return True
        
        return False
    
    # function to check if a dom is a double
    def double_check(self, dom):
        tuple = self.all_domino[dom]
        if tuple[0] == tuple[1]:
            return True
        return False


    # creates a list of hidden information by adding the opponent's hand back into the queue
    # then generate a cloned gameState with the opponents hand generated from the shuffled
    # unknown list
    def CloneAndRandomize(self):
        unknown = deepcopy(self.queue)  # create a deep copy of the queue

        for i in range(PLAYER_COUNT):
            if i != self.playerTurn:
                for dom in self.hands[i]: # put all of the opponent's dominoes in with the rest of the unknown dominoes
                    unknown.append(dom)

        new_hands = [[] for player in range(PLAYER_COUNT)]

        for dom in self.hands[self.playerTurn]:   # copy over the current players hand
            new_hands[self.playerTurn].append(dom)

        np.random.shuffle(unknown)

        for i in range(PLAYER_COUNT):
            if i != self.playerTurn:
                for k in range(len(self.hands[i])):
                    new_hands[i].append(unknown.pop())

        return GameState(new_hands, deepcopy(self.trains), unknown, self.playerTurn, self.passed)



    # converts the state to a (2 * player_count + 3)x28 binary representation 
    # (current_player's hand, size of each other player's hand, each player's train, mexican train, marked train indices, available heads to play on)
    def _binary(self):  # TODO signify multiples of a single head value being available

        state = np.zeros((2 * PLAYER_COUNT + 3, 28), dtype=np.int)
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

    def get_public_info(self, root = False):
        public_id = 'Turn: ' + str(self.playerTurn)

        if root:
            public_id += '|' + str([self.all_domino[dom] for dom in self.hands[self.playerTurn]])
            public_id += '\n'

        for i in range(PLAYER_COUNT):
            public_id += '|' + str(len(self.hands[i]))
        for train in self.trains:
            public_id += '|' + train.get_string()
        
        return public_id

    def _checkForEndGame(self):  # returns 1 if any player has an empty hand else 0 or all players have passed
        for hand in self.hands:
            if len(hand) == 0:
                return 1
        if False not in self.passed:
            """print(self.hands)
            print(self.passed)
            print("Train 0 head: {0}, marked: {1}".format(self.trains[0].head, self.trains[0].marked))
            print("Train 1 head: {0}, marked: {1}".format(self.trains[1].head, self.trains[1].marked))
            print("Train mex head: {0}, marked: {1}".format(self.trains[2].head, self.trains[2].marked))
            exit(0)"""
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
                winner = int(np.argmin(totals))

                temp = []
                for i in range(PLAYER_COUNT):
                    if i == winner:
                        temp.append(1)
                    else:
                        temp.append(-1)
                
                return temp
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

    # creates a copy of the current board with the players hands swapped, makes the chosen action, creates a new gameState
    # then returns the new gameState as well as it's value and an indication of the game being over or not
    def takeAction(self, action):   # TODO: add second turn after playing double
        new_hands = deepcopy(self.hands)

        new_trains = deepcopy(self.trains)

        next_player = (self.playerTurn + 1) % PLAYER_COUNT

        if action != -1:
            chosen_dom = action % 28
            try:
                new_hands[self.playerTurn].remove(chosen_dom) # remove played domino from current player's hand
            except:
                print("illegal action given to game state")
                print("chosen dom: {0}".format(chosen_dom))
                print("active player's hand: {0}".format(new_hands[self.playerTurn]))
                print("all hands: {0}".format(new_hands))
                exit(0)

            chosen_train = int(action/28)

            if chosen_train == PLAYER_COUNT:    # if chosen_train is equal to PLAYER_COUNT it is the mexican train
                new_trains[PLAYER_COUNT].add(chosen_dom)
            else:
                new_trains[(self.playerTurn + chosen_train) % PLAYER_COUNT].add(chosen_dom)

            if chosen_train == 0: # if the player played a domino on their own train unmark it (even if it isn't marked)
                new_trains[self.playerTurn].unmark()
            
            double_played = self.double_check(chosen_dom)
            # if the player played double they go again
            if double_played:
                next_player = self.playerTurn

            # mark any unfinished trains unless the unfinished train belongs to the current player
            # and the player played a double on it this turn
            for i, train in enumerate(new_trains):
                if train.unfinished and not (i == self.playerTurn and double_played and chosen_train == 0):
                    train.mark()

        newState = GameState(new_hands, new_trains, deepcopy(self.queue), next_player, self.passed)  # create new state

        return (newState, newState.value, newState.isEndGame)

    def render(self, logger):  # this logs each gamestate to a logfile in the run folder. The commented sections will print the game states to the terminal if you uncomment them
        logger.info("Current Turn: {0}".format(self.playerTurn))

        logger.info("Hands:\n{0}".format([[self.all_domino[dom] for dom in hand] for hand in self.hands]))


        for i in range(PLAYER_COUNT):
            logger.info("Train {0} head: {1}, Marked: {2}".format(i, self.trains[(self.playerTurn + i) % PLAYER_COUNT].head, self.trains[(self.playerTurn + i) % PLAYER_COUNT].marked))

        logger.info("Mexican Train: {0}".format(self.trains[PLAYER_COUNT].head))


        logger.info("Available actions (action #, train #, domino): {0}".format([(action, int(action/28), self.all_domino[action % 28]) for action in self.allowedActions]))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        logger.info('--------------')
        # print('--------------')
    def user_print(self):
        print("Hands:\n{0}".format([[self.all_domino[dom] for dom in hand] for hand in self.hands]))
        print("Your hand: {0}".format([self.all_domino[dom] for dom in self.hands[self.playerTurn]]))


        for i in range(PLAYER_COUNT):
            print("Train {0} head: {1}, Marked: {2}".format(i, self.trains[(self.playerTurn + i) % PLAYER_COUNT].head, self.trains[(self.playerTurn + i) % PLAYER_COUNT].marked))

        print("Mexican Train: {0}".format(self.trains[PLAYER_COUNT].head))


        print("Available actions (action #, train #, domino): {0}".format([(action, int(action/28), self.all_domino[action % 28]) for action in self.allowedActions]))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        print('--------------')

class Train:
    def __init__(self, first_dom, marked=True):
        self.all_domino = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        self.head_values = {0: 0, 2: 1, 5: 2, 9: 3, 14: 4, 20: 5,
                            27: 6}
        self.head_indices = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14, 5: 20, 6: 27}
        self.doms = [first_dom]
        self.head = self.head_values[first_dom]
        self.marked = marked
        self.unfinished = False

    def add(self, dom):
        self.doms.append(dom)
        tup = self.all_domino[dom]

        if tup[0] == tup[1]:
            self.unfinish()
        else:
            self.finish()

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

    def finish(self):
        self.unfinished = False
    
    def unfinish(self):
        self.unfinished = True
    
    def get_binary(self):
        b = np.zeros(28, dtype = np.int)
        b[self.doms] = 1
        return b
    
    def get_string(self):
        sorted_doms = sorted(self.doms)
        return str(sorted_doms) + ', Head: ' + str(self.head) + ', ' + str(self.marked)
