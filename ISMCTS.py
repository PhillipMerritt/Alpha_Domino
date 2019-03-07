# original source for this code: https://gist.github.com/kjlubick/8ea239ede6a026a61f4d

# to-do:
# create a clone and randomize function in game or gamestate class
# it will create a deep clone of the game state and randomize any 
# information hidden to the player

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """
    def __init__(self, move = None, parent = None, playerJustMoved = None):
        self.move = move # the move that brought us here
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1 # availablity count
        self.playerJustMoved = playerJustMoved

    def GetUntriedMoves(self, legalMoves):
        # Return the elements of legalMoves for which this node does not have children.

        # Find all moves for which this node DOES have children
        triedMoves = [child.move for child in self.childNodes]

        return [move for move in legalMoves if move not in triedMoves]

    def UCBSelectChild(self, legalMoves, exploration = 0.7):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
            exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """

        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if child.move in legalMoves]

        s = max(legalChildren, key = lambda c: float(c.wins)/float(c.visits) + exploration * sqrt(log(c.avails)/float(c.visits)))

        # Update availability counts
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
        """ Updaet this node - increment visit count by one, and increase win count by the result of the terminalState for self.playerJustMoved
        """
        self.visits += 1

        if self.playerJustMoved is not None:
            self.wins += terminalState.GetResult(self.playerJustMoved)

    def __repr__(self):
        return "[M:%s W/V/A: %4i/%4i/%4i]" % (self.move, self.wins, self.visits, self.avails)

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

    for i in range(itermax):
        node = rootnode

        # Determinize
        state = rootstate.CloneAndRandomize(rootstate.playerToMove)

        # Select
        untriedMoves = node.GetUntriedMoves(state.GetMoves())
        if untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(untriedMoves)
            player = state.playerToMove
            state.DoMove(m)
            node = node.AddChild(m, player) # add child and descend

        # Simulate
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropogate from the expanded node and work back to the root node
            node.Update(state)
            node = node.parentNode

    # Output some info about the tree if verbose is true
    if (verbose): print rootnode.TreeToString(0)
    else: print rootnode.ChildrenToString()

    return max(rootnode.childNodes, key = lambda c: c.visits).move # return the move that was most visited