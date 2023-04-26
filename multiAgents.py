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

        "*** YOUR CODE HERE ***"

        score=0
        for i in range (len(newGhostStates)):
            pos=newGhostStates[i].getPosition()
            ghostdist=manhattanDistance(newPos, pos)
            if ghostdist<=2:
                score-=2-ghostdist

        foodpos=newFood.asList()
        fdist=[]
        for f in foodpos:
            fooddist=manhattanDistance(newPos,f)
            fdist.append(fooddist)
        if (len(fdist) != 0):
            if min(fdist) == 0:
                score += 1
            else:
                score += 1/min(fdist)




        return successorGameState.getScore()+score






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

        nextact = 0
        curval = float("-inf") #returning the max value, so should start from small
        acts = gameState.getLegalActions(0)


        def maxValue(gameState,agentIndex, depth):
            maxval = float("-inf") #returning the max value, so should start from small
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                succ = gameState.generateSuccessor(agentIndex, action) #working on successor states
                check=eval(succ, agentIndex, depth)
                maxval = max(maxval, check)
            return maxval

        def minValue(gameState,agentIndex, depth):
            minval = float("inf") #returning minumum value so should start from largest
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                succ=gameState.generateSuccessor(agentIndex, action)
                check=eval(succ, agentIndex, depth)
                minval = min(minval, check)
            return minval


        def eval(gameState,agentIndex, depth):
            #in a recursive call with other functions
            change=gameState.getNumAgents()
            if change-1<=agentIndex: #checking for pacman
                depth += 1
                agentIndex = 0
            else:
                agentIndex += 1

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()

            if self.depth<=depth:
                return gameState.getScore()
            #recursively calls max and min for pacman and ghosts until reaches a loss or win
            if agentIndex == 0:
                return maxValue(gameState, agentIndex,depth)
            else:
                return minValue(gameState, agentIndex, depth)



        for act in acts:
            succ=gameState.generateSuccessor(0, act)
            check = eval(succ, 0, 0)
            if check > curval:
                curval = check
                nextact = act
        return nextact





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #works similar as the minmaxagent, alpha and beta added for alpha beta pruning.
        nextact = 0
        curval = float("-inf")
        acts = gameState.getLegalActions(0)
        alpha=float("-inf")
        beta=float("inf")

        def maxValue(gameState, agentIndex, depth, alpha, beta):
            maxval = float("-inf")
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                succ = gameState.generateSuccessor(agentIndex, action)
                check=eval(succ, agentIndex, depth, alpha, beta)
                maxval = max(maxval,check)
                if (maxval>beta):
                    return maxval
                alpha=max(alpha,maxval)
            return maxval

        def minValue(gameState, agentIndex, depth, alpha, beta):
            minval = float("inf")
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                succ = gameState.generateSuccessor(agentIndex, action)
                check= eval(succ, agentIndex, depth,alpha,beta)
                minval = min(minval, check )
                if (minval<alpha):
                    return minval
                beta=min(minval,beta)
            return minval

        def eval(gameState, agentIndex, depth, alpha, beta):
            change = gameState.getNumAgents()
            if change-1<=agentIndex:
                depth += 1
                agentIndex = 0
            else:
                agentIndex += 1

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()

            if self.depth<=depth:
                return gameState.getScore()

            if agentIndex == 0:
                return maxValue(gameState, agentIndex, depth, alpha,beta)
            else:
                return minValue(gameState, agentIndex, depth, alpha,beta)

        for act in acts:
            succ=gameState.generateSuccessor(0, act)
            check = eval(succ, 0, 0, alpha,beta)
            if check > curval:
                curval = check
                nextact = act
            alpha=max(alpha,curval)
        return nextact


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
#works in a similar way with minimax agent but this time instead of min, running against ghost chooses random actions among the set.
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        nextact = 0
        curval = float("-inf")
        acts = gameState.getLegalActions(0)

        def maxValue(gameState, agentIndex, depth):
            maxval = float("-inf")
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                succ = gameState.generateSuccessor(agentIndex, action)
                check = eval(succ, agentIndex, depth)
                maxval = max(maxval, check)
            return maxval

        def eval(gameState, agentIndex, depth):
                change = gameState.getNumAgents()
                if change - 1 <= agentIndex:
                    depth += 1
                    agentIndex = 0
                else:
                    agentIndex += 1

                if gameState.isWin() or gameState.isLose():
                    return gameState.getScore()

                if self.depth <= depth:
                    return gameState.getScore()

                if agentIndex == 0:
                    return maxValue(gameState, agentIndex, depth)

                else:
                    l=len(gameState.getLegalActions(agentIndex)) #instead of min
                    val=0
                    for i in gameState.getLegalActions(agentIndex):
                        succ = gameState.generateSuccessor(agentIndex, i)
                        val+=eval(succ, agentIndex, depth)

                    return val/l

        for act in acts:
            succ = gameState.generateSuccessor(0, act)
            check = eval(succ, 0, 0)
            if check > curval:
                curval = check
                nextact = act
        return nextact




        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I did not change all variables to currentgamnestate instead I only changed in the first assignment of successor state which is for this function currentGameState
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    capsulepos=currentGameState.getCapsules()


    score = 0
    for i in range(len(newGhostStates)):
        pos = newGhostStates[i].getPosition()
        ghostdist = manhattanDistance(newPos, pos)
        if ghostdist <= 2:
            score -= 2 - ghostdist
        else:
            score-=ghostdist


    foodpos = newFood.asList()
    fdist = []
    for f in foodpos:
        fooddist = manhattanDistance(newPos, f)
        fdist.append(fooddist)
    if (len(fdist) != 0):
        if min(fdist) == 0:
            score += 1
        else:
            score += 1/min(fdist)



    for c in capsulepos:
        cdist=manhattanDistance(newPos,c)
        score+=1/cdist


    return successorGameState.getScore()


    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
