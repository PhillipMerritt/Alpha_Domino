import numpy as np
import logging
import globals
from globals import *
from config import PLAYER_COUNT
from copy import deepcopy
from collections import defaultdict
from random import randrange

# MEXICAN TRAIN
# Game generates GameStates and handles switching between player turns
class Game:

    def __init__(self):     #TODO: add a # of players parameter
        hands, trains, queue = self._generate_board()  # generate a new board and choose the starting player based on who has the highest double

        self.gameState = GameState(hands, trains, queue, self.currentPlayer, [GameState.Clues() for i in range(PLAYER_COUNT)])  # create a GameState
        self.actionSpace = [np.zeros(  # action space is DOM_COUNT * each train
            (DOM_COUNT * len(trains)), dtype=np.int)]
        #self.grid_shape = (4, DOM_COUNT)  # grid shape is 7xDOM_COUNT
        #self.input_shape = self.grid_shape = self.gameState.binary.shape  # input shape for the neural network is the shape of the binary state representation
        self.input_shape = self.grid_shape = (2 * PLAYER_COUNT + 3, DOM_COUNT)
        self.name = 'mexican_train'
        self.state_size = len(self.gameState.binary)  # size of the entire game state # TODO: look into this to see if it could be effecting anything

        self.action_size = [len(self.actionSpace[0])]  # size of the actionSpace

    def reset(self):  # creates new game
        #print("NEW GAME\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        count = 1
        while 1:    # sometimes game just play out to completion without any choices being made so games are created until this doesn't happen
            hands, trains, queue = self._generate_board()

            self.gameState = GameState(hands, trains, queue, self.currentPlayer, [GameState.Clues() for i in range(PLAYER_COUNT)])

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

            for i in range(HANDSIZE):
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

        for i, hand in enumerate(hands):
            for dom in hand:
                if dom in DOUBLES and dom > highest_doubles[i]:
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
    def __init__(self, hands, trains, queue, playerTurn, clues, passed = [False for player in range(PLAYER_COUNT)]):
        self.hands = hands
        self.trains = trains
        self.queue = queue
        self.clues = clues
        self.passed = passed

        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move

        self.playerTurn = playerTurn
        self.public_id = self.get_public_info()
        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now

        self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        if len(self.allowedActions) != 0:
            self.passed[self.playerTurn] = False
        
        self.decision_type = 0

    def _draw(self):  # draws a random domino then updates binary and id. If there are no more dominos to draw return false
        #print("failed to play on: {0}\nUnfinished: {1}\nHand: {2}".format(sorted(missing_doms), [train.head for train in self.trains if train.unfinished],[INDEX2TUP[dom] for dom in sorted(self.hands[self.playerTurn])]))

        if self.queue:  # if there are dominoes to draw
            self.hands[self.playerTurn].append(self.queue.pop())  # randomly pop one from the boneyard and place it in the players hand

            self.binary = self._binary()
            self.id = self._convertStateToId()

            return True

        return False

    # generates a list of all allowed actions. If there are no available actions dominoes are drawn if available
    # until there are actions to be made. The actions are in the form of action = (train num * DOM_COUNT) + action (ex. domino 14 to train 3 would be (3*DOM_COUNT)+14 = 98
    def _allowedActions(self):
        heads = defaultdict(list)
        available_pips = []
        unfinished = False

        # check to see if any trains have an unfinished double
        # players have to finish doubles if they can unless they
        # played the double the previous turn on a train that isn't theirs
        for i in range(PLAYER_COUNT):
            """if i < PLAYER_COUNT:
                index = (self.playerTurn + i) % PLAYER_COUNT
            else:
                index = i"""

            if self.trains[i].unfinished:
                unfinished = True
                # if the unfinished train belongs to this player and it isn't marked 
                # then they played the double last turn and must finish it
                if i == self.playerTurn and not self.trains[i].marked:
                    heads[self.trains[i].head].append(0)

                    if self.trains[i].head not in available_pips:
                        available_pips.append(self.trains[i].head)

                    break
                else:
                    if i > self.playerTurn:
                        offset = i - self.playerTurn
                    else:
                        offset = i + PLAYER_COUNT - self.playerTurn

                    heads[self.trains[i].head].append(offset)

                    if self.trains[i].head not in available_pips:
                        available_pips.append(self.trains[i].head)
        

        if not unfinished: # if the player isn't forced to play on specific trains due to doubles
            for i in range(PLAYER_COUNT):  # create a list of the available head values
                if i == self.playerTurn or self.trains[i].marked:
                    offset = 0
                    if i > self.playerTurn:
                        offset = i - self.playerTurn
                    elif i < self.playerTurn:
                        offset = i + PLAYER_COUNT - self.playerTurn

                    heads[self.trains[i].head].append(offset)

                    if self.trains[i].head not in available_pips:
                        available_pips.append(self.trains[i].head)
            
            heads[self.trains[PLAYER_COUNT].head].append(PLAYER_COUNT)    # mexican train

            if self.trains[PLAYER_COUNT].head not in available_pips:
                available_pips.append(self.trains[PLAYER_COUNT].head)
        
        

        # check for legal actions. If none found draw a domino and try again. If still none found pass turn and mark this train
        actions = []
        
        
        for dom_index in self.hands[self.playerTurn]: # for each domino in hand
            low_pip, high_pip = INDEX2TUP[dom_index]
            
            if heads[low_pip]:
                for offset in heads[low_pip]:
                    actions.append(offset * DOM_COUNT + dom_index)
                
            
            if low_pip != high_pip and heads[high_pip]:
                for offset in heads[high_pip]:
                    actions.append(offset * DOM_COUNT + dom_index)
                

            

        if actions:  # if there are any available actions return them
            return actions
        elif not self._draw():  # if no actions found draw a domino
            self.clues[self.playerTurn].draw(available_pips, True)
            self.passed[self.playerTurn] = True
            return []  # if drawing a domino fails return an empty list

        
        

        self.clues[self.playerTurn].draw(available_pips)
        

        new_dom = self.hands[self.playerTurn][-1]   # get the drawn domino
        low_pip, high_pip = INDEX2TUP[new_dom]
        if heads[low_pip]:
                for offset in heads[low_pip]:
                    actions.append(offset * DOM_COUNT + new_dom)
                
            
        if low_pip != high_pip and heads[high_pip]:
            for offset in heads[high_pip]:
                actions.append(offset * DOM_COUNT + new_dom)
        
        if len(actions) > 0:
            return actions
        
        
        
        self.trains[self.playerTurn].mark()
        self.clues[self.playerTurn].draw(available_pips, True)
        
        return []

    # function to determine if a domino can be played on the given head value
    def match_check(self, dom, head):
        return head == INDEX2TUP[dom][0] or head == INDEX2TUP[dom][1]
    
    # function to check if a dom is a double
    def double_check(self, dom):
        tuple = INDEX2TUP[dom]
        if tuple[0] == tuple[1]:
            return True
        return False


    # creates a list of hidden information by adding the opponent's hand back into the queue
    # then generate a cloned gameState with the opponents hand generated from the shuffled
    # unknown list
    def CloneAndRandomize(self, count):
        if PLAYER_COUNT == 2 and not self.queue:
            return GameState(list(self.hands), deepcopy(self.trains), [], self.playerTurn, deepcopy(self.clues), self.passed)
        
        states = []
        perm_unknown = deepcopy(self.queue)  # create a deep copy of the queue
        hidden_players = []

        og_allowed = [[] for _ in range(PLAYER_COUNT)]
        
        for i in range(PLAYER_COUNT):
            if i != self.playerTurn:
                hidden_players.append(i)
                perm_unknown.extend(self.hands[i])
                self.clues[i].reset()
                og_allowed[i] = deepcopy(self.clues[i].allowed)
        
        for _ in range(count):
            unknown = deepcopy(perm_unknown)
            np.random.shuffle(unknown)

            for i in range(PLAYER_COUNT):
                if i != self.playerTurn:
                    self.clues[i].reset(og_allowed[i])

            while len(unknown) != len(self.queue):  #TODO: find where dominos are being added to hands but not taken out of unknown
                for i, player in enumerate(hidden_players):
                    if len(self.clues[player].hand) == len(self.hands[player]):
                        continue

                    drawn_dom = self.clues[player].random_draw(unknown)
                    if drawn_dom == None:
                        stolen = False

                        if len(hidden_players) > 1:
                            #index = i + random.randint(1,len(hidden_players) - 1)
                            temp_players = list(hidden_players)
                            np.random.shuffle(temp_players)
                            for other_player in temp_players:
                                if other_player == player:
                                    continue
                                if self.clues[player].steal(self.clues[other_player]):
                                    stolen = True
                                    break

                        # swap w/ boneyard
                        if not stolen:
                            pre = len(self.clues[player].hand)
                            if not self.clues[player].boneyard_swap(unknown):
                                #print("BONEYARD SWAP FAIL")
                                if pre != len(self.clues[player].hand):
                                    print("lost one in boneyard swap")
                                    quit(0)
                                unknown.extend(self.clues[player].hand)
                                self.clues[player].reset()
                    elif drawn_dom != None and drawn_dom in unknown:
                        print("error in random_draw function")
                        quit(0)

            new_hands = [[] for i in range(PLAYER_COUNT)]
            for player in hidden_players:
                new_hands[player] = list(self.clues[player].hand)

            # copy over the current players hand
            new_hands[self.playerTurn] = list(self.hands[self.playerTurn])

            yield GameState(new_hands, deepcopy(self.trains), unknown, self.playerTurn, deepcopy(self.clues), self.passed)

    def random_insert(self, lst, item):
        lst.insert(randrange(len(lst)+1), item)

    # converts the state to a (2 * player_count + 3)xDOM_COUNT binary representation 
    # (current_player's hand, size of each other player's hand, each player's train, mexican train, marked train indices, available heads to play on)
    def _binary(self):  # TODO signify multiples of a single head value being available

        state = np.zeros((2 * PLAYER_COUNT + 3, DOM_COUNT), dtype=np.int)
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
            public_id += '|' + str([INDEX2TUP[dom] for dom in self.hands[self.playerTurn]])
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
                totals = [sum([sum(INDEX2TUP[dom]) for dom in hand]) for hand in self.hands]
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
    def takeAction(self, action):
        new_hands = deepcopy(self.hands)

        new_trains = deepcopy(self.trains)

        new_clues = deepcopy(self.clues)
        

        next_player = (self.playerTurn + 1) % PLAYER_COUNT

        if action != -1:
            chosen_dom = action % DOM_COUNT
            new_hands[self.playerTurn].remove(chosen_dom)
            """try:
                new_hands[self.playerTurn].remove(chosen_dom) # remove played domino from current player's hand
            except:
                print("illegal action given to game state")
                print("chosen dom: {0}".format(chosen_dom))
                print("active player's hand: {0}".format(new_hands[self.playerTurn]))
                print("all hands: {0}".format(new_hands))
                exit(0)"""

            low_pip, high_pip = INDEX2TUP[chosen_dom]
            clues = self.clues[self.playerTurn].clue_dict
            if (clues[low_pip] and clues[low_pip][0] == 0) or (clues[high_pip] and clues[high_pip][0] == 0): 
                self.user_print()
                print(INDEX2TUP[chosen_dom])
                print(str(self.clues[self.playerTurn]))
                quit(0)

            # update any relevant clues
            new_clues[self.playerTurn].play(chosen_dom)

            chosen_train = int(action/DOM_COUNT)

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

        newState = GameState(new_hands, new_trains, deepcopy(self.queue), next_player, new_clues, list(self.passed))  # create new state

        return (newState, newState.value, newState.isEndGame)

    def render(self, logger):  # this logs each gamestate to a logfile in the run folder. The commented sections will print the game states to the terminal if you uncomment them
        logger.info("Current Turn: {0}".format(self.playerTurn))

        logger.info("Hands:\n{0}".format([[INDEX2TUP[dom] for dom in hand] for hand in self.hands]))


        for i in range(PLAYER_COUNT):
            logger.info("Train {0} head: {1}, Marked: {2}".format(i, self.trains[(self.playerTurn + i) % PLAYER_COUNT].head, self.trains[(self.playerTurn + i) % PLAYER_COUNT].marked))

        logger.info("Mexican Train: {0}".format(self.trains[PLAYER_COUNT].head))


        logger.info("Available actions (action #, train #, domino): {0}".format([(action, int(action/DOM_COUNT), INDEX2TUP[action % DOM_COUNT]) for action in self.allowedActions]))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        logger.info('--------------')
        # print('--------------')
    def user_print(self):
        print("Hands:\n{0}".format([[INDEX2TUP[dom] for dom in hand] for hand in self.hands]))
        print("Your hand: {0}".format([INDEX2TUP[dom] for dom in self.hands[self.playerTurn]]))


        for i in range(PLAYER_COUNT):
            print("Train {0} head: {1}, Marked: {2}".format(i, self.trains[(self.playerTurn + i) % PLAYER_COUNT].head, self.trains[(self.playerTurn + i) % PLAYER_COUNT].marked))

        print("Mexican Train: {0}".format(self.trains[PLAYER_COUNT].head))


        print("Available actions (action #, train #, domino): {0}".format([(action, int(action/DOM_COUNT), INDEX2TUP[action % DOM_COUNT]) for action in self.allowedActions]))
        # print("Dominoes left in boneyard: {0}".format(np.count_nonzero(self.board[3])))

        print('--------------')
    
    class Clues():
        def __init__(self):
            #self.clue_arr = []
            self.clue_dict = defaultdict(list)
            self.hand = []
            self.counts = defaultdict(int)
            self.allowed = [True for i in range(len(INDEX2TUP))]
            self.og_allowed = []

        # sets the maximum count for each pip value that the player
        # failed to play on. Then since the player just drew increment
        # the maximum count for each tracked pip value by 1.
        # If second_attempt is true then they still couldn't play a 
        # domino after drawing so all maximum values for the missing_pips
        # stay zero instead.
        def draw(self, missing_pips, second_attempt = False):
            for pip in missing_pips:    # mark maximum count for each pip value that the player failed to play on as zero
                self.clue_dict[pip] = [0]

            if not second_attempt:  # if they just drew a domino then they can have one more of each tracked pip value
                for pip in self.clue_dict.keys():
                    if self.clue_dict[pip]:
                        self.clue_dict[pip][0] += 1

        # if either pip values of the played domino are being tracked decrement the total # the player could have
        def play(self, dom):
            low_pip, high_pip = INDEX2TUP[dom]

            """if self.clue_dict[low_pip] or self.clue_dict[high_pip]:
                for i in range(len(self.clue_arr[self.playerTurn])):
                    if self.clue_arr[i][0] == low_pip:
                        self.clue_arr[i][1] -= 1

                        self.clue_dict[low_pip] = self.clue_dict[i][1]

                        if not self.clue_dict[high_pip] or low_pip == high_pip:
                            break
                    elif self.clue_arr[i][0] == high_pip:
                        self.clue_arr[i][1] -= 1

                        self.clue_dict[high_pip] = self.clue_dict[i][1]
                        break"""
            
            if self.clue_dict[low_pip]:
                if self.clue_dict[low_pip][0] == 0:
                    print(self.clue_dict)
                    print(INDEX2TUP[dom])
                    exit(0)
                self.clue_dict[low_pip][0] -= 1
                if not self.clue_dict[low_pip][0]:
                    for i in PIP2INDICES[low_pip]:
                        self.allowed[i] = False
            
            if self.clue_dict[high_pip] and low_pip != high_pip:
                if self.clue_dict[high_pip][0] == 0:
                    print(self.clue_dict)
                    print(INDEX2TUP[dom])
                    exit(0)
                self.clue_dict[high_pip][0] -= 1
                if not self.clue_dict[high_pip][0]:
                    for i in PIP2INDICES[high_pip]:
                        self.allowed[i] = False
        
        def approve(self, dom):
            return self.allowed[dom]

        # attempts to draw an allowed domino from the passed unknown list
        def random_draw(self, unknown):
            rejects = []    # rejected draws will be set aside to prevent them from being drawn again

            while unknown:  # until all dominos in unknown have been rejected
                dom = unknown.pop()

                if self.approve(dom):   # successful draw
                    low_pip, high_pip = INDEX2TUP[dom]

                    self.counts[low_pip] += 1   # increment the counter for that pip value
                    if self.clue_dict[low_pip] and self.counts[low_pip] == self.clue_dict[low_pip][0]:  # if that pip value is being tracked and the max # has been reached
                        for disallowed in PIP2INDICES[low_pip]: # disallow all dominos that contain that pip value
                            self.allowed[disallowed] = False
                    
                    if low_pip != high_pip: # only repeat this process for the higher pip value if this isn't a double
                        self.counts[high_pip] += 1
                        if self.clue_dict[high_pip] and self.counts[high_pip] == self.clue_dict[high_pip][0]:
                            for disallowed in PIP2INDICES[high_pip]:
                                self.allowed[disallowed] = False
                    
                    unknown.extend(rejects) # put the rejects back into the unknown list
                    self.hand.append(dom)   # add the approved dom to hand and return it
                    return dom
                else:
                    rejects.append(dom)
            
            # if no dominos were drawn then they were all rejected so put them back and return None
            unknown.extend(rejects)
            return None
        
        # swap dominos w/ the boneyard until an additional domino can be drawn
        def boneyard_swap(self, boneyard):
            unattempted = []   #TODO: only need to set aside dominos with tracked values
            for dom in self.hand:
                low_pip, high_pip = INDEX2TUP[dom]
                if self.clue_dict[low_pip] or self.clue_dict[high_pip]:
                    unattempted.append(dom)

            og_len = len(boneyard)
            while 1:
                if not unattempted: # if setting aside every domino from the hand didn't allow two draws return false
                    return False

                # choose a random untried domino to set aside
                set_aside = np.random.choice(unattempted)

                # remove it from hand
                self.hand.remove(set_aside)
                unattempted.remove(set_aside)

                # update the approval list after it is taken out
                self.update_approval(set_aside)
                
                # attempt the first draw
                draw_a = self.random_draw(boneyard)
                if draw_a != None:
                    # if successful attempt the second
                    draw_b = self.random_draw(boneyard)
                    if draw_b != None:
                        # if succesful put the domino that was set aside back into the boneyard, update boneyard, and return True
                        #temp_yard.append(set_aside)
                        
                        """print("Boneyard swap")
                        print(sorted(og_hand))
                        print(sorted(self.hand))
                        print(sorted(boneyard))
                        print(sorted(temp_yard))"""
                        #boneyard = deepcopy(temp_yard)
                        boneyard.append(set_aside)
                        return True
                    else:
                        # if the first draw was successful, but not the second
                        # remove the first draw from hand and update approval 
                        self.update_approval(draw_a)
                        self.hand.remove(draw_a)
                        boneyard.append(draw_a)

                # setting aside this domino didn't work so put it back in hand
                # random_draw is used for this so that the approvals are updated
                self.random_draw([set_aside])



            
        
        def steal(self, other_player):
            """potential_offerings = [dom for dom in self.hand f other_player.approve(dom)]
            potential_recieving = []"""
            #temp_hand = list(self.hand)
            np.random.shuffle(other_player.hand)
            recieved = self.random_draw(other_player.hand)

            if not recieved:
                return False
            
            other_player.update_approval(recieved)

            """given = other_player.random_draw(temp_hand)

            if not given:
                self.hand.remove(recieved)
                other_player.random_draw([recieved])
                return False
            
            self.hand.remove(given)
            self.update_approval(given)"""

            return True

            

        # updates the approval list after a domino has been given away
        def update_approval(self, dom):
            low_pip, high_pip = INDEX2TUP[dom]
            self.allowed[dom] = True

            self.counts[low_pip] -= 1
            if low_pip != high_pip:
                self.counts[high_pip] -= 1

            if self.clue_dict[low_pip] and self.counts[low_pip] == self.clue_dict[low_pip][0] - 1: # this pip was disallowed
                # check each domino with this pip value to see if it can be allowed again
                for disallowed in PIP2INDICES[low_pip]:
                    if disallowed != dom:
                        # get the other pip value
                        tup = INDEX2TUP[disallowed]

                        if tup[0] == tup[1]:    # if the disallowed domino is the double of this pip value just allow it
                            self.allowed[disallowed] = True
                            continue
                        elif tup[0] != low_pip:
                            other_pip = tup[0]
                        else:
                            other_pip = tup[1]
                        
                        if not self.clue_dict[other_pip] or self.counts[other_pip] != self.clue_dict[other_pip][0]:
                            self.allowed[disallowed] = True

            if low_pip != high_pip:
                if self.clue_dict[high_pip] and self.counts[high_pip] == self.clue_dict[high_pip][0] - 1: # this pip was disallowed
                    # check each domino with this pip value to see if it can be allowed again
                    for disallowed in PIP2INDICES[high_pip]:
                        if disallowed != dom:
                            # get the other pip value
                            tup = INDEX2TUP[disallowed]

                            if tup[0] == tup[1]:    # if the disallowed domino is the double of this pip value just allow it
                                self.allowed[disallowed] = True
                                continue
                            elif tup[0] != high_pip:
                                other_pip = tup[0]
                            else:
                                other_pip = tup[1]
                            
                            if not self.clue_dict[other_pip] or self.counts[other_pip] != self.clue_dict[other_pip][0]:
                                self.allowed[disallowed] = True
            
        def reset(self, og_allowed = None):
            self.hand = []
            self.counts = defaultdict(int)
            if og_allowed:
                self.allowed = deepcopy(og_allowed)
            else:
                self.allowed = [True for i in range(len(INDEX2TUP))]

                for i, doms in enumerate(PIP2INDICES):
                    if self.clue_dict[i] and not self.clue_dict[i][0]:
                        for dom in doms:
                            self.allowed[dom] = False
        
        def __str__(self):
            return str(self.clue_dict)


class Train:
    def __init__(self, first_dom, marked=True):
        self.doms = [first_dom]
        self.head = HEAD_VALUES[first_dom]
        self.marked = marked
        self.unfinished = False

    def add(self, dom):
        self.doms.append(dom)
        tup = INDEX2TUP[dom]

        if tup[0] == tup[1]:
            self.unfinish()
        else:
            self.finish()

        if tup[0] == self.head:
            self.head = tup[1]
        else:
            self.head = tup[0]

    def match(self, dom):
        tup = INDEX2TUP[dom]
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
        b = np.zeros(DOM_COUNT, dtype = np.int)
        b[self.doms] = 1
        return b
    
    def get_string(self):
        sorted_doms = sorted(self.doms)
        return str(sorted_doms) + ', Head: ' + str(self.head) + ', ' + str(self.marked)
