from math import sqrt, log
import random

from timeit import default_timer as timer

class Node:
    def __init__(self, move = None, parent = None, playerJustMoved = None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = playerJustMoved
    
    def GetUntriedMoves(self, legalMoves):
        triedMoves = [child.move for child in self.childNodes]

        return [move for move in legalMoves if move not in triedMoves]
        
    def UCBSelectChild(self, legalMoves, exploration = 0.7):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
            exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """
        
        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if child.move in legalMoves]
        
        # Get the child with the highest UCB score
        s = max(legalChildren, key = lambda c: float(c.wins)/float(c.visits) + exploration * sqrt(log(c.avails)/float(c.visits)))
        
        # Update availability counts -- it is easier to do this now than during backpropagation
        for child in legalChildren:
            child.avails += 1
        
        # Return the child selected above
        return s

    def AddChild(self, m, p):
        """ Add a new child node for the move m.
            Return the added child node
        """
        n = Node(move = m, parent = self, playerJustMoved = p)
        self.childNodes.append(n)
        return n
    
    def Update(self, terminalState):
        """ Update this node - increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved.
        """
        self.visits += 1
        if self.playerJustMoved is not None:
            self.wins += terminalState.value[self.playerJustMoved]

    def __repr__(self):
        return "[M:{} W/V/A: {:4}/{:4}/{:4}]".format(self.move, self.wins, self.visits, self.avails)

    def TreeToString(self, indent):
        """ Represent the tree as a string, for debugging purposes.
        """
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s
    
    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s
    
    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s

def ISMCTS(rootstate, itermax, verbose = False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
    """


    rootnode = Node()
    randomized_state_generator = rootstate.CloneAndRandomize(itermax)
    
    # Determinize at start of each loop
    for state in randomized_state_generator:
        
        node = rootnode
        
        start = timer()
        
        # Select
        while not state.isEndGame and node.GetUntriedMoves(state.allowedActions) == []: # node is fully expanded and non-terminal
            node = node.UCBSelectChild(state.allowedActions)
            state, _, _ = state.takeAction(node.move)
            
            while not state.isEndGame and len(state.allowedActions) < 2:
                if state.allowedActions == []:
                    state, _, _ = state.takeAction(-1)
                else:
                    state, _, _ = state.takeAction(state.allowedActions[0])

        # Expand
        untriedMoves = node.GetUntriedMoves(state.allowedActions)
        if untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(untriedMoves) 
            player = state.playerTurn
            state, _, _ = state.takeAction(m)
            node = node.AddChild(m, player) # add child and descend tree

        # Simulate
        while not state.isEndGame: # while state is non-terminal
            if state.allowedActions == []:
                state, _, _ = state.takeAction(-1)
            elif len(state.allowedActions) == 1:
                state, _, _ = state.takeAction(state.allowedActions[0])
            else:
                state, _, _ = state.takeAction(random.choice(state.allowedActions))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state)
            node = node.parentNode
        

    # Output some information about the tree - can be omitted
    """if (verbose): print rootnode.TreeToString(0)
    else: print rootnode.ChildrenToString()"""

    #print("Time to run ismcts: {}, time getting moves: {}, extra actions taken: {}".format(total_time, untried_time, actions_taken))
    
    return max(rootnode.childNodes, key = lambda c: c.visits).move # return the move that was most visited