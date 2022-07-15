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
import random
import util
import sys

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()

        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2', a=0, b=0, c=0, d=0, e=0, f=0, g=0):
        self.index = 0  # Pacman is always agent index 0
        # self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        # global a1, a2, a3, a4, a5, a6, a7
        # a1, a2, a3, a4, a5, a6, a7 = float(a), float(b), float(
        #     c), float(d), float(e), float(f), float(g)
        # with open("weight.txt", "a") as myfile:
        #     myfile.write("\nThe coeffcients " + str(a) + " " + str(b) +
        #                  " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) + " " + str(g))
        self.evaluationFunction = util.lookup(evalFn, globals())


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agentIdx, depth):
            numAgents = state.getNumAgents()
            agentIdx = agentIdx % numAgents

            if (state.isLose() or state.isWin()):
                return self.evaluationFunction(state), None
            if agentIdx == 0:
                return maxValue(state, agentIdx, depth)
            else:
                return minValue(state, agentIdx, depth)

        def maxValue(state, agentIdx, depth):
            depth = depth + 1
            if depth > self.depth:
                return self.evaluationFunction(state), None

            bestScore, bestActions = -(sys.maxsize), None
            successorActions = state.getLegalActions(agentIdx)
            for succAction in successorActions:
                successors = []
                successors.append(state.generateSuccessor(
                    agentIdx, succAction))
                for successor in successors:
                    bestCurrent = minimax(
                        successor, agentIdx + 1, depth)[0]
                    if bestScore is None:
                        bestScore = bestCurrent
                        bestActions = succAction
                    else:
                        if bestScore < bestCurrent:
                            bestScore = bestCurrent
                            bestActions = succAction

            return bestScore, bestActions

        def minValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state), None

            bestScore, bestActions = sys.maxsize, None
            successorActions = state.getLegalActions(agentIdx)
            for succAction in successorActions:
                successors = []
                successors.append(state.generateSuccessor(
                    agentIdx, succAction))
                for successor in successors:
                    bestCurrent = minimax(
                        successor, agentIdx + 1, depth)[0]
                    if bestScore is None:
                        bestScore = bestCurrent
                        bestActions = succAction
                    else:
                        if bestScore > bestCurrent:
                            bestScore = bestCurrent
                            bestActions = succAction

            return bestScore, bestActions

        bestScore, bestAction = minimax(gameState, 0, 0)
        return bestAction

        """
        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                # value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(
                    agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(
                    agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)
        return action
        """

        """
        def minimax_search(state, agentIndex, depth):
            # if in min layer and last ghost
            if agentIndex == state.getNumAgents():
                # if reached max depth, evaluate state
                if depth == self.depth:
                    return self.evaluationFunction(state)
                # otherwise start new max layer with bigger depth
                else:
                    return minimax_search(state, 0, depth + 1)
            # if not min layer and last ghost
            else:
                moves = state.getLegalActions(agentIndex)
                # if nothing can be done, evaluate the state
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                # get all the minimax values for the next layer with each node being a possible state after a move
                next = (minimax_search(state.generateSuccessor(
                    agentIndex, m), agentIndex + 1, depth) for m in moves)

                # if max layer, return max of layer below
                if agentIndex == 0:
                    return max(next)
                # if min layer, return min of layer below
                else:
                    return min(next)
        # select the action with the greatest minimax value
        result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(
            gameState.generateSuccessor(0, x), 1, 1))

        return result
        """


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def AlphaBetaPruning(state, agentIdx, depth, alpha, beta):
            numAgents = state.getNumAgents()
            agentIdx = agentIdx % numAgents

            if (state.isLose() or state.isWin()):
                return self.evaluationFunction(state), None
            if agentIdx == 0:
                return maxValue(state, agentIdx, depth, alpha, beta)
            else:
                return minValue(state, agentIdx, depth, alpha, beta)

        def maxValue(state, agentIdx, depth, alpha, beta):
            depth = depth + 1
            if depth > self.depth:
                return self.evaluationFunction(state), None

            bestScore, bestActions = -(sys.maxsize), None
            successorActions = state.getLegalActions(agentIdx)
            for succAction in successorActions:
                successors = []
                successors.append(state.generateSuccessor(
                    agentIdx, succAction))
                for successor in successors:
                    bestcurrent = AlphaBetaPruning(
                        successor, agentIdx + 1, depth, alpha, beta)[0]
                    if bestScore is None:
                        bestScore = bestcurrent
                        bestActions = succAction
                    else:
                        if bestScore < bestcurrent:
                            bestScore = bestcurrent
                            bestActions = succAction
                    if bestScore > beta:
                        return bestScore, bestActions
                    alpha = max(alpha, bestScore)
            return bestScore, bestActions

        def minValue(state, agentIdx, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state), None

            bestScore, bestActions = sys.maxsize, None
            successorActions = state.getLegalActions(agentIdx)
            for succAction in successorActions:
                successors = []
                successors.append(state.generateSuccessor(
                    agentIdx, succAction))
                for successor in successors:
                    bestCurrent = AlphaBetaPruning(
                        successor, agentIdx + 1, depth, alpha, beta)[0]
                    if bestScore is None:
                        bestScore = bestCurrent
                        bestActions = succAction
                    else:
                        if bestScore > bestCurrent:
                            bestScore = bestCurrent
                            bestActions = succAction
                    if bestScore < alpha:
                        return bestScore, bestActions
                    beta = min(bestScore, beta)
            return bestScore, bestActions

        alpha = -(sys.maxsize)
        beta = sys.maxsize
        bestScore, bestAction = AlphaBetaPruning(gameState, 0, 0, alpha, beta)
        return bestAction


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
        "*** YOUR CODE HERE ***"

        def Expectimax(state, agentIdx, depth):
            numAgents = state.getNumAgents()
            agentIdx = agentIdx % numAgents

            if (state.isLose() or state.isWin()):
                return self.evaluationFunction(state), None
            if agentIdx == 0:
                return maxValue(state, agentIdx, depth)
            else:
                return expectimaxVal(state, agentIdx, depth)

        def maxValue(state, agentIdx, depth):
            depth = depth + 1
            if depth > self.depth:
                return self.evaluationFunction(state), None

            bestScore = -(sys.maxsize)
            bestActions = None
            successorActions = state.getLegalActions(agentIdx)
            for succAction in successorActions:
                successors = []
                successors.append(state.generateSuccessor(
                    agentIdx, succAction))
                for successor in successors:
                    bestcurrent = Expectimax(
                        successor, agentIdx + 1, depth)[0]
                    if bestScore < bestcurrent:
                        bestScore = bestcurrent
                        bestActions = succAction
            return bestScore, bestActions

        def expectimaxVal(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state), None

            bestScore = 0
            bestActions = None
            successorActions = state.getLegalActions(agentIdx)
            for succAction in successorActions:
                successors = []
                successors.append(state.generateSuccessor(
                    agentIdx, succAction))
                for successor in successors:
                    p = (1 / len(successors))
                    bestScore += p * \
                        (Expectimax(successor, agentIdx + 1, depth)[0])
                    bestActions = succAction
            return bestScore, bestActions

        bestScore, bestAction = Expectimax(gameState, 0, 0)
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")

    pacman_pos = currentGameState.getPacmanPosition()
    newCapsules = currentGameState.getCapsules()
    capsules_left = len(currentGameState.getCapsules())
    all_food = currentGameState.getFood().asList()
    food_left = len(all_food)
    md_closest_food = min([util.manhattanDistance(
        pacman_pos, food) for food in all_food])

    if newCapsules:
        closestCapsule = min([manhattanDistance(pacman_pos, caps)
                             for caps in newCapsules])
    else:
        closestCapsule = 0

    scared_ghost, active_ghost = [], []
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer:
            scared_ghost.append(ghost)
        else:
            active_ghost.append(ghost)

    dist_nearest_scaredghost = dist_nearest_activeghost = 0

    if not len(scared_ghost):
        dist_nearest_scaredghost = 0

    if not len(active_ghost):
        dist_nearest_activeghost = float("inf")

    if active_ghost:
        dist_nearest_activeghost = min([util.manhattanDistance(
            pacman_pos, ghost.getPosition()) for ghost in active_ghost])
        if dist_nearest_activeghost > 10:
            dist_nearest_activeghost = 10
    if scared_ghost:
        dist_nearest_scaredghost = min([util.manhattanDistance(
            pacman_pos, ghost.getPosition()) for ghost in scared_ghost])

    ans = currentGameState.getScore() + -5*md_closest_food + -5*(1.0/dist_nearest_activeghost) + \
        0*(dist_nearest_scaredghost) + \
        -4*capsules_left + -4*food_left + 3*closestCapsule

    # ans = a1*currentGameState.getScore() + a2*md_closest_food + a3*(1.0/dist_nearest_activeghost) + \
    #     a4*dist_nearest_scaredghost + a5*capsules_left + a6*food_left + a7*closestCapsule

    return ans

    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    closestGhost = min([manhattanDistance(newPos, ghost.getPosition())
                       for ghost in newGhostStates])
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps)
                             for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food)
                          for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule
    """


# Abbreviation
better = betterEvaluationFunction
