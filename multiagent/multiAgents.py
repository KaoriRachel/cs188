# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
            
        for ghostState in newGhostStates:
            if manhattanDistance(ghostState.configuration.getPosition(), newPos) <= 1.1:
                return -float('inf')

        curNum = currentGameState.getNumFood();
        newNum = successorGameState.getNumFood();
        if newNum < curNum:
            return 0

        total = float('inf')
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    total = min(manhattanDistance(newPos, [i, j]), total)
        return -total

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        #ghost from 1 to num-1
        num = gameState.getNumAgents()

        def max_value_action(state, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(0)
            # Choose one of the best actions
            max_scores = -float('inf')
            for action in legalMoves:
                newState = state.generateSuccessor(0, action)
                max_scores = max(max_scores, min_value_action(newState, 1, depth))
            return max_scores

        #layers means the index of the current agent
        def min_value_action(state, layers, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(layers)
            # Choose one of the best actions
            min_scores = float('inf')
 
            for action in legalMoves:
                newState = state.generateSuccessor(layers, action)
                if layers == num - 1:
                    if depth == 0:
                        min_scores = min(min_scores, self.evaluationFunction(newState))
                    else:
                        min_scores = min(min_scores, max_value_action(newState, depth - 1))
                else:
                    min_scores = min(min_scores, min_value_action(newState, layers + 1, depth))
            return min_scores

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)
        # Choose one of the best actions
        max_scores = -float('inf')
        for action in legalMoves:
            v = min_value_action(gameState.generateSuccessor(0, action), 1, self.depth - 1)
            if v > max_scores:
                max_scores, max_action = v, action
        return max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        #ghost from 1 to num-1
        num = gameState.getNumAgents()

        def max_value_action(state, depth, alpha, beta):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(0)
            # Choose one of the best actions
            max_scores = -float('inf')
            for action in legalMoves:
                newState = state.generateSuccessor(0, action)
                max_scores = max(max_scores, min_value_action(newState, 1, depth, alpha, beta))
                if max_scores > beta:
                    return max_scores
                alpha = max(alpha, max_scores)
            return max_scores

        #layers means the index of the current agent
        def min_value_action(state, layers, depth, alpha, beta):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(layers)
            # Choose one of the best actions
            min_scores = float('inf')
 
            for action in legalMoves:
                newState = state.generateSuccessor(layers, action)
                if layers == num - 1:
                    if depth == 0:
                        min_scores = min(min_scores, self.evaluationFunction(newState))
                    else:
                        min_scores = min(min_scores, max_value_action(newState, depth - 1, alpha, beta))
                else:
                    min_scores = min(min_scores, min_value_action(newState, layers + 1, depth, alpha, beta))
                if min_scores < alpha:
                    return min_scores
                beta = min(beta, min_scores)
            return min_scores

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)
        # Choose one of the best actions
        max_scores, alpha, beta = -float('inf'), -float('inf'), float('inf')
        for action in legalMoves:
            v = min_value_action(gameState.generateSuccessor(0, action), 1, self.depth - 1, alpha, beta)
            if v > max_scores:
                max_scores, max_action = v, action
                if max_scores > beta:
                    return max_scores
                alpha = max(alpha, max_scores)
        return max_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        #ghost from 1 to num-1
        num = gameState.getNumAgents()

        def max_value_action(state, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(0)
            # Choose one of the best actions
            max_scores = -float('inf')
            for action in legalMoves:
                newState = state.generateSuccessor(0, action)
                max_scores = max(max_scores, random_value_action(newState, 1, depth))
            return max_scores

        #layers means the index of the current agent
        def random_value_action(state, layers, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(layers)
            # Choose one of the best actions
            random_scores, actionNum = 0, 0
 
            for action in legalMoves:
                actionNum += 1
                newState = state.generateSuccessor(layers, action)
                if layers == num - 1:
                    if depth == 0:
                        random_scores += self.evaluationFunction(newState)
                    else:
                        random_scores += max_value_action(newState, depth - 1)
                else:
                    random_scores += random_value_action(newState, layers + 1, depth)
            return random_scores / actionNum

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)
        # Choose one of the best actions
        max_scores = -float('inf')
        for action in legalMoves:
            v = random_value_action(gameState.generateSuccessor(0, action), 1, self.depth - 1)
            if v >= max_scores:
                max_scores, max_action = v, action
        return max_action
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    singleFoodMark = 100;
    numMarks = singleFoodMark * currentGameState.getNumFood();
    if numMarks == 0:
        return 0
    pos = currentGameState.getPacmanPosition()
    for ghostState in currentGameState.getGhostStates():
        if manhattanDistance(ghostState.configuration.getPosition(), pos) <= 1.01:
            return -float('inf')

    newFood = currentGameState.getFood()
    min_dist = float('inf')
    for i in range(newFood.width):
        for j in range(newFood.height):
            if newFood[i][j] and manhattanDistance(pos, [i, j]) < min_dist:
                min_dist = manhattanDistance(pos, [i, j])
                min_pos = [i, j]

    def numToSign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    walls = currentGameState.getWalls()
    wallsNum = 0
    if walls[pos[0] + numToSign(min_pos[0] - pos[0])][pos[1]]:
        wallsNum += 1.1
    if walls[pos[0]][pos[1] + numToSign(min_pos[1] - pos[1])]:
        wallsNum += 1.1 
    #if walls[pos[0] + numToSign(min_pos[0] - pos[0])][pos[1] + numToSign(min_pos[1] - pos[1])]:
    #    wallsNum += 0.2
    return -(numMarks + min_dist + wallsNum)

# Abbreviation
better = betterEvaluationFunction
