import numpy as np
import logging
import globals
from copy import deepcopy
from collections import defaultdict

# all_domino is a list of tuples containing the pip value for each domino
INDEX2TUP = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3), (0, 4),
                           (1, 4), (2, 4), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6),
                           (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
DOM_COUNT = len(INDEX2TUP)
PLAYER_COUNT = 4
MAX_PIP = 6
HANDSIZE = DOM_COUNT / PLAYER_COUNT

DOUBLES = [i for i in range(DOM_COUNT) if INDEX2TUP[i][0] == INDEX2TUP[i][1]]
HONOR_POINTS = {8:5,11:5,15:5,20:10,25:10}
HONORS = set([8, 11, 15, 20, 25])
PIP2INDICES = {0:set([0]),1:set([1,2]),2:set([3,4,5]),3:set([6,7,8,9]),4:set([10,11,12,13,14]),5:set([15,16,17,18,19,20]),6:set([21,22,23,24,25,26,27,28]),7:set([0, 2, 5, 9, 14, 20, 27])}

POSSIBLE_BIDS = [0]
POSSIBLE_BIDS.extend(list(range(30,43)))
#POSSIBLE_BIDS.extend([84,168,336,672,1344,2688])

# when evaluating a domino if you input the suit as the index to dom_rank and the index identity of the domino as the key (second index)
# you will get the ranking of that domino in that suit (1-7, 7 being the highest rank)
# ex. comparing (1,4) to (3:4)
# ranking of (1,4) = dom_rank[4][11] which gives 1
# ranking of (4,4) = dom_rank[4][14] which gives 7
# (4,4) > (1,4)
# each suit also includes dominoes that would normally be in other suits unless the lower pip value is the trump suit
DOM_RANK = [defaultdict(int, {0:7,1:1,3:2,6:3,10:4,15:5,21:6}), defaultdict(int, {1:1,2:7,4:2,7:3,11:4,16:5,22:6}), defaultdict(int, {3:1,4:2,5:7,8:3,12:4,17:5,23:6}), defaultdict(int, {6:1,7:2,8:3,9:7,13:4,18:5,24:6})
                , defaultdict(int, {10:1,11:2,12:3,13:4,14:7,19:5,25:6}), defaultdict(int, {15:1,16:2,17:3,18:4,19:5,20:7,26:6}), defaultdict(int, {21:1,22:2,23:3,24:4,25:5,26:6,27:7}), defaultdict(int, {0:1, 2:2, 5:3, 9:4, 14:5, 20:6, 27:7})]

def get_all_ranks(dom):
    ranks = set()
    low_pip, high_pip = INDEX2TUP[dom]
    ranks.add(low_pip)
    ranks.add(high_pip)
    
    if low_pip == high_pip:
        ranks.add(7)
    
    return ranks
    
RANK_SETS = [get_all_ranks(dom) for dom in range(DOM_COUNT)]

BIDS_TO_MARKS = {84:2,168:3,336:4,672:5,1344:6,2688:7} # dict to look up how many marks are won by winning a trick

MARKS_TO_WIN = 3

# Game generates GameStates and handles switching between player turns
class Game:

    def __init__(self):
        hands, collections, passed, marks, tricks_won = self._generate_board()  # generate a new board

        self.gameState = GameState(np.random.random_integers(4) - 1, hands, None, collections, -1, -1, passed, marks, tricks_won, -1, -1, -1)  # create a GameState
        # action space:
        # 20 for bids 30-42, 84, 168, 336, 672, 1344, 2688, pass
        # 28 for each domino
        # 9 for choosing trump suit (suits 0-6, doubles, follow me)
        #self.actionSpace = [np.zeros((28), dtype=np.int), np.zeros((9),dtype=np.int)]
        self.grid_shape = self.input_shape = self.gameState.binary.shape
        self.name = 'Texas_42'
        self.state_size = len(self.gameState.binary)  # size of the entire game state
        #self.action_size = [len(space) for space in self.actionSpace]  # size of the actionSpace

    def reset(self):  # sets player to 1 and generates a new board and gamestate
        hands, collections, passed, marks, tricks_won = self._generate_board()

        self.gameState = GameState(np.random.random_integers(4) - 1, hands, None, collections, -1, -1, passed, marks, tricks_won, -1, -1, -1)

        return self.gameState

    def _generate_board(self):
        hands = [[], [], [], []]
        queue = globals.queue_reset()  # reset and shuffle the queue

        for i in range(PLAYER_COUNT):
            index = i*7
            hands[i] = queue[index : index + 7]

        for i, hand in enumerate(hands):  # whoever has the 0:1 domino bids first
            if hands[i][0] == 1 or hands[i][1] == 1:
                self.currentPlayer = i


        collections = [[],[]]
        passed = [False,False,False,False]
        marks = [0, 0]
        tricks_won = [0, 0]

        return hands, collections, passed, marks, tricks_won

    # once an action has been chosen this function is called and it keeps making actions until their is a choice to be made
    def step(self, action, logger=None, user=False):  # game state makes the chosen action and returns the next state, the value of the next state for the active player in that state, and whether or not the game is over
        while 1:
            next_state, value, done = self.gameState.takeAction(
                action)  # value is always 0 unless the game is over. Otherwise it is -1 since the last player made a winning move
            self.gameState = next_state  # updates the gameState
            self.currentPlayer = next_state.playerTurn  # swaps current player

            info = None  # idk what this is
            if logger:
                self.gameState.render(logger,action)  # I moved rendering to here so that the automated turns would still be logged
            if done or len(self.gameState.allowedActions) > 1:  # if the game is over or the current player has a choice break the loop
                break
            elif len(self.gameState.allowedActions) == 1:  # else takeAction() with the one action available
                action = self.gameState.allowedActions[0]

                if user:                        # if a user is playing print the automated turn
                    self.gameState.user_print()
            else:  # or if no actions are available pass turn by taking action -1
                print("No available actions!")

        return ((next_state, value, done, info))

    def identities(self, state, actionValues):  # haven't looked into what this function is doing quite yet
        identities = [(state, actionValues)]

        return identities


class GameState():
    def __init__(self, playerTurn, hands, clues, collections, high_bid, highest_bidder, passed,
                 marks, tricks_won, trump_suit, f_m_suit, decision_type, played_dominoes=[-1,-1,-1,-1]):

        self.hands = hands
        self.clues = clues
        self.played_dominoes = played_dominoes
        self.collections = collections
        self.tricks_won = tricks_won
        self.marks = marks

        self.trump_suit = trump_suit    # 0, 1, 2, 3, 4, 5, 6, 7, 8 (7 = doubles) (8 = follow me)
        self.fm_suit = f_m_suit

        self.passed = passed
        self.high_bid = high_bid
        self.highest_bidder = highest_bidder

        self.decision_type = decision_type

        self.playerTurn = playerTurn

        self.isEndGame = self._checkForEndGame()

        if self.isEndGame:
            self.allowedActions = []
        else:
            self.allowedActions = self._allowedActions()  # generates the list of possible actions that are then given to the neural network

        self.binary = self._binary()  # this is a binary representation of the board state which is basically just the board atm
        #self.id = self._convertStateToId()  # the state ID is all 4 board lists appended one after the other.
        # these previous two may have been converted poorly from connect4 and are causing issues now

        self.value = self._getValue()  # the value is from the POV of the current player. So either 0 for the game continuing or -1 if the last player made a winning move
        #self.score = self._getScore()

    def _allowedActions(self):
        # The player holding the 0-1 tile bids first. Each player may either bid or pass. Each bid must be a number from 30 to 42, and must raise the previous bid. If the bid is maxed out (42), it may be doubled (84), and then doubled again (168). Bids of 42 or greater are made only my taking all 42 points.
        if self.decision_type == 0: # a domino is to be played
            if self.trump_suit == -1 or self.fm_suit == -1: # this the leading player
                actions = list(self.hands[self.playerTurn])
            else:   # else this is a following player
                actions = [dom for dom in self.hands[self.playerTurn] if self.get_suit(dom)] # dominoes that are in the follow or trump suit are available actions

                if len(actions) == 0:  # if the player can't follow suit they can play anything from their hand
                    self.clues[self.playerTurn].add(self.trump_suit)
                    self.clues[self.playerTurn].add(self.fm_suit)
                    actions = list(self.hands[self.playerTurn])

        elif self.decision_type == 1:   # a pip or doubles is to be chosen as the trump suit alternatively the player can pass
            low_pip, high_pip = INDEX2TUP[self.played_dominoes[self.playerTurn]]
            
            if low_pip == high_pip: # if a double is lead the trump can be the double's suit, doubles, or follow me
                actions = [low_pip, 7]
            else:                   # otherwise it can be either pip value or follow me
                actions = [low_pip, high_pip]
                
        else:   # bidding phase
            def filter_bids(bid):
                if bid > 42:
                    if bid == self.high_bid * 2:
                        return True
                    return False  
                elif bid > self.high_bid or bid == 0:
                    return True
                return False
            
            actions = list(filter(filter_bids, POSSIBLE_BIDS))

        return actions

    # creates a list of hidden information by adding the opponent's hand back into the queue
    # then generate a cloned gameState with the opponents hand generated from the shuffled
    # unknown list
    def CloneAndRandomize(self, yield_count):
        def approve_dom(clues, dom):
            dom_ranks = RANK_SETS[dom]
            
            return len(clues.intersection(dom_ranks)) == 0
        
        def assignADomino(unassignedDominoes, players, valid_doms, hands, hand_sizes):
            if not players:
                return hands
            
            p_i = players.pop()
            players = [p_i] + players
            p = valid_doms[p_i]
            
            for d in unassignedDominoes:
                if d not in p:
                    continue
                
                valid_hands = makeAssignment(d, p_i, list(unassignedDominoes), list(players), [v.copy() for v in valid_doms], [list(h) for h in hands], hand_sizes)
                if valid_hands:
                    return valid_hands

            return None # backtrack
        
        def makeAssignment(d, p_i, unassignedDominoes, players, valid_doms, hands, hand_sizes):
            baseMakeAssignment(d, p_i, unassignedDominoes, players, valid_doms, hands, hand_sizes) # pass by reference
            if not makeMandatoryAssignments(unassignedDominoes, players, valid_doms, hands, hand_sizes):
                return None
            return assignADomino(unassignedDominoes, players, valid_doms, hands, hand_sizes)
        
        def baseMakeAssignment(d, p_i, unassignedDominoes, players, valid_doms, hands, hand_sizes):
            hands[p_i].append(d)
            unassignedDominoes.remove(d)

            if len(hands[p_i]) >= hand_sizes[p_i]:
                players.remove(p_i)
            
            for player in players:
                valid_doms[player].discard(d)
                
        def makeMandatoryAssignments(unassignedDominoes, players, valid_doms, hands, hand_sizes):
            for p_i in list(players):
                possible = len(hands[p_i]) + len(valid_doms[p_i])
                if  possible > hand_sizes[p_i]:
                    continue
                elif possible < hand_sizes[p_i]:
                    return False
                
                req = list(valid_doms[p_i])
                valid_doms[p_i] = set()
                for d in req:
                    baseMakeAssignment(d, p_i, unassignedDominoes, players, valid_doms, hands, hand_sizes) # pass by ref
                
                return makeMandatoryAssignments(unassignedDominoes, players, valid_doms, hands, hand_sizes)
            
            return True  # success  
            
        optimize = self.clues != None and True in [len(c) > 0 for c in self.clues]
        optimize = False
        
        players = [i for i in range(PLAYER_COUNT) if i != self.playerTurn and len(self.hands[i]) != 0]
                
        hand_sizes = [len(self.hands[i]) for i in range(PLAYER_COUNT)]
        
        og_unknown = []
        for i in players:
            og_unknown.extend(self.hands[i])
        
        super_og_unknown = list(og_unknown)
        
        
        hands = [[], [], [], []]
        
        if optimize:
            valid_doms = [set([dom for dom in og_unknown if approve_dom(self.clues[i], dom)]) if i in players else set() for i in range(PLAYER_COUNT)]
            og_valid_doms = [set([dom for dom in og_unknown if approve_dom(self.clues[i], dom)]) if i in players else set() for i in range(PLAYER_COUNT)]
            og_unknown = set(og_unknown)
            
            another = True
            while another:
                remove = []
                another = False
                # look for any players with only one possible hand
                for p in players:
                    if len(valid_doms[p]) == hand_sizes[p]:
                        hands[p] = list(valid_doms[p])
                        remove.append(p)
                        og_unknown = og_unknown.difference(valid_doms[p])
                        
                        for i in players:
                            if i in remove:
                                continue
                            valid_doms[i] = valid_doms[i].difference(valid_doms[p])
                        
                        valid_doms[p] = set()
                
                for p in remove:
                    players.remove(p)
                    another = True
                
            another = True
            while another:
                another = False        
                remove = []
                unique = [valid_doms[0].difference(valid_doms[1]|valid_doms[2]|valid_doms[3]), valid_doms[1].difference(valid_doms[0]|valid_doms[2]|valid_doms[3]),
                        valid_doms[2].difference(valid_doms[1]|valid_doms[0]|valid_doms[3]), valid_doms[3].difference(valid_doms[1]|valid_doms[2]|valid_doms[0])]
                for p in players:
                    s = unique[p]
                    
                    if not s:
                        continue
                    
                    hands[p].extend(list(s))
                    og_unknown = og_unknown.difference(s)
                    
                    if hand_sizes[p] == len(hands[p]):
                        remove.append(p)
                        valid_doms[p] = set()
                    else:
                        valid_doms[p] = valid_doms[p].difference(s)
                
                for p in remove:
                    players.remove(p)
                    another = True
            
            players = sorted(players, key=lambda i: len(valid_doms[i]) - len(self.hands[i]) - len(hands[i]))
            og_unknown = list(og_unknown)
            
            if False not in [valid_doms[players[0]] == valid_doms[p] for p in players[1:]]:
                optimize = False
              
        for _ in range(yield_count):
            unknown = list(og_unknown)
            np.random.shuffle(unknown)
            
            if optimize:
                assert True not in [len(valid_doms[p]) == 0 for p in players], "set error"
                fails = 0    
                while 1:
                    new_hands = assignADomino(unknown, list(players), [v.copy() for v in valid_doms], [list(h) for h in hands], hand_sizes)

                    if not new_hands:
                        fails += 1
                        np.random.shuffle(players)
                    else:
                        break
                assert False not in [set(new_hands[i]).issubset(og_valid_doms[i]) for i in range(4) if i != self.playerTurn], "Bad dom"
                assert False not in [len(self.hands[p]) == len(new_hands[p]) for p in players], "Hand size mismatch"
            else:
                new_hands = [list(h) for h in hands]
                
                for i in players:
                    for j in range(hand_sizes[i] - len(hands[i])):
                        new_hands[i].append(unknown.pop())
                        
            new_hands[self.playerTurn] = list(self.hands[self.playerTurn])

            yield GameState(self.playerTurn, new_hands, [c.copy() for c in self.clues] if self.clues else None, [list(self.collections[0]), list(self.collections[1])], self.high_bid, self.highest_bidder, list(self.passed), list(self.marks),
                            list(self.tricks_won), self.trump_suit, self.fm_suit, self.decision_type, list(self.played_dominoes))

    def _binary(self):  #TODO: maybe add passed
        def encode_doms(binary, doms):
            for dom in doms:
                binary[INDEX2TUP[dom]] = 1
            
                
        position = np.zeros((7,8,9), dtype=np.int)
        
        if self.decision_type != -1:
            encode_doms(position[0], [dom for dom in self.played_dominoes if dom != -1])
        position[0][-2][self.trump_suit] = 1
        position[0][-1][self.fm_suit] = 1
                
        for i, hand in enumerate(self.hands):
            index = i + 1
            encode_doms(position[index], hand)
            
            if i % 2 == self.highest_bidder:
                position[index][-2] = 1
            
            if i == self.playerTurn:
                position[index][-1] = 1

        encode_doms(position[-2], self.collections[0])
        position[-2][-2] = self.marks[0]
        position[-2][-1] = self.tricks_won[0]
        
        encode_doms(position[-1], self.collections[1])
        position[-1][-2] = self.marks[1]
        position[-1][-1] = self.tricks_won[1]

        return position

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

        for turn in turn_sequence:  # played domino for this trick for each player
            id += '|' + str(self.played_dominoes[turn])

        return id
    
    def get_public_info(self, root = False):
        id = 'Trump: ' + str(self.trump_suit) + ', Follow: ' + str(self.fm_suit) + '\n'

        if root:
            id += str([INDEX2TUP[dom] for dom in self.hands[self.playerTurn]])
            id += '\n'

        for i, dom in enumerate(self.played_dominoes):
            if dom == -1 and i == self.playerTurn:
                id += '|A'
            elif dom == -1:
                id += '|-'
            else:
                id += '|' + str(INDEX2TUP[dom])

        return id

    def _checkForEndGame(self):
        return self.marks[0] == MARKS_TO_WIN or self.marks[1] == MARKS_TO_WIN

    def _getValue(self):
         # This is the value of the state
        if self.marks[0] == MARKS_TO_WIN:
            return [1,-1]
        elif self.marks[1] == MARKS_TO_WIN:
            return [-1, 1]
        else:
            return [0,0]

    def deal_hands(self):
        hands = [[], [], [], []]
        queue = globals.queue_reset()  # reset and shuffle the queue

        for i in range(PLAYER_COUNT):
            index = i*7
            hands[i] = queue[index : index + 7]

        return hands

    # returns the suit of a domino and a bool indicating if it is a trump 
    # suit can be 0-6 or 7 if it is a double and doubles are the trump suit
    def get_suit(self, dom):
        ranks = RANK_SETS[dom]
        if self.trump_suit in ranks or self.fm_suit in ranks:
            return True 

    # finds out the winner of the trick and adds the honors of the trick to that team's collection
    def trick_score(self, played_dominoes):
        new_collections = [list(self.collections[0]), list(self.collections[1])]
        weights = [DOM_RANK[self.trump_suit][dom] for dom in self.played_dominoes]

        if weights == [0, 0, 0, 0]:
            weights = [DOM_RANK[self.fm_suit][dom] for dom in self.played_dominoes]
        
        winning_player = np.argmax(weights)

        #print("T: {0}, FM: {1}".format(self.trump_suit, self.fm_suit))
        #print("played doms: {0}, winning player: {1}".format(doms, winning_player))

        winning_team = winning_player % 2

        # honors played this trick go to the winner of the trick
        new_collections[winning_team].extend([dom for dom in played_dominoes if dom in HONORS])

        new_tricks_won = list(self.tricks_won)
        new_tricks_won[winning_team] += 1

        return new_tricks_won, new_collections, winning_player

    def score_collections(self,collections):
        points = [0,0]
        for i, col in enumerate(collections):
            for dom in col:
                points[i]+=HONOR_POINTS[dom]

        return points

    def takeAction(self, action):
        #if action not in self.allowedActions:
        #    print("Illegal Action!")
        #    exit()

        # play domino action
        if self.decision_type == 0:
            new_hands = [list(hand) for hand in self.hands]            
            new_collections = [list(self.collections[0]), list(self.collections[1])]

            new_hands[self.playerTurn].remove(action)   # remove played dom from hand

            new_played = list(self.played_dominoes)
            new_played[self.playerTurn] = action        # add played dom to played_dominoes

            if self.trump_suit == -1: #TODO: look into passed
                newState = GameState(self.playerTurn, new_hands, None, new_collections, self.high_bid, self.highest_bidder, list(self.passed),
                                      list(self.marks), list(self.tricks_won), -1, -1, 1, new_played) 
            elif self.fm_suit == -1:
                newState = GameState((self.playerTurn+1)%4, new_hands, [c.copy() for c in self.clues], new_collections, self.high_bid, self.highest_bidder,
                                     list(self.passed), list(self.marks), list(self.tricks_won), self.trump_suit, INDEX2TUP[action][1], 0, new_played)
            elif -1 not in new_played:    # trick is over
                new_tricks_won, new_collections, winner = self.trick_score(new_played)  # score trick to get new trick win count, collections and winner
                added_points = self.score_collections(new_collections)  # score collections
                added_points[0] += new_tricks_won[0]    # add 1 point for each trick won to each team's points
                added_points[1] += new_tricks_won[1]
                new_marks = list(self.marks)
                hand_complete = False

                # points_to_win is how many points the bidding team needs to win the mark(s)
                # and points_to_block is how much the other team needs to prevent this
                if self.high_bid >= 42: 
                    points_to_win = 42
                    points_to_block = 1
                else:
                    points_to_win = self.high_bid
                    points_to_block = 43 - self.high_bid

                # if the highest_bidder is team 0 and they scored enough to win or if the highest_bidder is team 1 and team 0 scored enough to block
                if (self.highest_bidder == 0 and added_points[0] >= points_to_win) or (self.highest_bidder == 1 and added_points[0] >= points_to_block):
                    hand_complete = True

                    # team 0 wins the mark(s)
                    if self.high_bid <= 42:
                        new_marks[0] += 1
                    else:
                        new_marks[0] += BIDS_TO_MARKS[self.high_bid]
                # if the highest_bidder is team 1 and they scored enough to win or if the highest_bidder is team 0 and team 1 scored enough to block
                elif (self.highest_bidder == 1 and added_points[1] >= points_to_win) or (self.highest_bidder == 0 and added_points[1] >= points_to_block):
                    hand_complete = True

                    # team 1 wins the mark(s)
                    if self.high_bid <= 42:
                        new_marks[1] += 1
                    else:
                        new_marks[1] += BIDS_TO_MARKS[self.high_bid]

                if hand_complete: # if the trick is over and anyone has an empty hand this hand is over
                    new_passed = [False, False, False, False]
                    new_collections = [[],[]]
                    new_tricks_won = [0,0]
                    new_hands = self.deal_hands()
                    newState = GameState(winner, new_hands, None, new_collections, -1, -1, new_passed, new_marks, new_tricks_won, -1, -1, -1)
                else:
                    newState = GameState(winner, new_hands, [c.copy() for c in self.clues], new_collections, self.high_bid, self.highest_bidder,
                                         list(self.passed), list(self.marks), new_tricks_won, self.trump_suit, -1, 0)
            else:
                newState = GameState((self.playerTurn+1)%4, new_hands, [c.copy() for c in self.clues], new_collections, self.high_bid, self.highest_bidder,
                                     list(self.passed), list(self.marks), list(self.tricks_won), self.trump_suit, self.fm_suit, 0, new_played)
        elif self.decision_type == 1:   # the player that won the bid chooses the trump suit for the hand after playing their first dom
            new_trump_suit = action

            newState = GameState((self.playerTurn+1)%4, [list(hand) for hand in self.hands], [set(), set(), set(), set()], self.collections, self.high_bid, self.highest_bidder,
                                 self.passed, self.marks, self.tricks_won, new_trump_suit, new_trump_suit, 0, list(self.played_dominoes))
        else:   # bid action # TODO: Make bidding only only one bid from each player
            new_hands = [list(hand) for hand in self.hands] # make copies of needed info      
            new_high_bid = self.high_bid
            new_passed = list(self.passed)
            highest_bidder = self.highest_bidder
            
            if action == 0:                         # player passes 
                new_passed[self.playerTurn] = True
            else:
                new_high_bid = action
                highest_bidder = self.playerTurn

            pass_count = 0              # count # of players who have passed
            for p in new_passed:
                if p:
                    pass_count += 1

            if pass_count >= 3 and highest_bidder != -1:         # if 3/4 players have passed
                new_passed = [True, True, True, True]
                next_d_type = 0                     # move onto playing the first domino
                next_player = highest_bidder        # next player will be the winning bidder
                highest_bidder = highest_bidder % 2 # changed to team index rather than player after bidding phase
            elif pass_count == 4 and highest_bidder == -1:   # if no one cast a bid shuffle the hands and start this hand over
                new_passed = [False, False, False, False]
                next_d_type = -1
                next_player = (self.playerTurn + 1) % 4
                new_hands = self.deal_hands()
            else:                       # else continue bidding
                next_d_type = -1
                for i in range(1,5):
                    if not new_passed[(self.playerTurn + i) % 4]:
                        next_player = (self.playerTurn + i) % 4
                        break

            newState = GameState(next_player, new_hands, None, [[],[]], new_high_bid, highest_bidder, new_passed, self.marks, self.tricks_won, -1, -1,
                         next_d_type)  # create new state

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
            temp_hand = []

            for index in self.allowedActions:
                temp_hand.append((index, INDEX2TUP[index]))

            logger.info("Available Actions: {0}".format(temp_hand))
        else:
            logger.info("Available Actions: {0}".format(self.allowedActions))

        logger.info("Current Marks: Team 0 = {0}/7, Team 1 = {1}/7".format(self.marks[0],self.marks[1]))
        logger.info("Tricks Won: Team 0 = {0}/7, Team 1 = {1}/7".format(self.tricks_won[0],self.tricks_won[1]))

        # logger.info("{0}".format())
        temp_hand = []
        temp_hand_2 = []

        for index in self.collections[0]:
            temp_hand.append(index)
        for index in self.collections[1]:
            temp_hand_2.append(index)

        logger.info("Team 0 Collection: {0} | Team 1 Collection: {1}".format(temp_hand,temp_hand_2))

        if self.decision_type == -1:
            logger.info("Passed: {0}".format(self.passed))

        for i,turn in enumerate(turn_sequence):
            temp_hand = []
            if i == 0:
                logger.info("Team 0 Hands:")
            elif i == 2:
                logger.info("Team 1 Hands:")

            for index in self.hands[turn]:
                temp_hand.append(INDEX2TUP[index])

            if turn == self.playerTurn:
                logger.info("Player {0}: {1} <<<< Active Player".format(turn, temp_hand))
            else:
                logger.info("Player {0}: {1}".format(turn, temp_hand))

        if self.high_bid > 0:
            logger.info("Highest Bidder: {0} Highest Bid: {1}".format(self.highest_bidder,self.high_bid))

        if self.decision_type != -1:
            logger.info("Trump Suit: {0}".format(self.trump_suit))
            logger.info("FM Suit: {0}".format(self.fm_suit))

            for i, turn in enumerate(turn_sequence):
                if i == 0:
                    logger.info("Team 0 Played Dominoes:")
                elif i == 2:
                    logger.info("Team 1 Played Dominoes:")
                if self.played_dominoes[turn] == -1:
                    dom = 'None'
                else:
                    dom = str(INDEX2TUP[self.played_dominoes[turn]])

                logger.info("Player {0}: {1}".format(turn,dom))

        logger.info('--------------')

    def user_print(self):
        turn_sequence = [0, 2, 1, 3]

        print('--------------')
        print("Current Turn: {0} | DECISION TYPE: {1}".format(self.playerTurn, self.decision_type))
        print("Current Marks: Team 0 = {0}/7, Team 1 = {1}/7".format(self.marks[0], self.marks[1]))
        print("Tricks Won: Team 0 = {0}/7, Team 1 = {1}/7".format(self.tricks_won[0], self.tricks_won[1]))

        # print("{0}".format())
        temp_hand = []
        temp_hand_2 = []

        for index in self.collections[0]:
            temp_hand.append(index)
        for index in self.collections[1]:
            temp_hand_2.append(index)

        print("Team 0 Collection: {0} | Team 1 Collection: {1}".format(temp_hand, temp_hand_2))

        if self.decision_type == -1:
            print("Passed: {0}".format(self.passed))

        for i, turn in enumerate(turn_sequence):
            temp_hand = []
            if i == 0:
                print("Team 0 Hands:")
            elif i == 2:
                print("Team 1 Hands:")

            for index in self.hands[turn]:
                temp_hand.append(INDEX2TUP[index])

            if turn == self.playerTurn:
                print("Player {0}: {1} <<<< Active Player".format(turn, temp_hand))
            else:
                print("Player {0}: {1}".format(turn, temp_hand))

        if self.high_bid > 0:
            print("Highest Bidder: {0} Highest Bid: {1}".format(self.highest_bidder, self.high_bid))

        if self.decision_type != -1:
            print("Trump Suit: {0}".format(self.trump_suit))
            print("FM Suit: {0}".format(self.fm_suit))


            for i, turn in enumerate(turn_sequence):
                if i == 0:
                    print("Team 0 Played Dominoes:")
                elif i == 2:
                    print("Team 1 Played Dominoes:")

                if self.played_dominoes[turn] != -1:
                    print("Player {0}: {1}".format(turn, INDEX2TUP[self.played_dominoes[turn]]))
                else:
                    print("Player {0}: None".format(turn))
        if self.decision_type == -1:
            print("Available Actions: {0}".format(self.allowedActions))
        elif self.decision_type == 0:
            temp_hand = []

            for index in self.allowedActions:
                temp_hand.append((index, INDEX2TUP[index]))

            print("Available Actions: {0}".format(temp_hand))
        else:
            print("Available Actions: {0}".format(self.allowedActions))
        print('--------------')