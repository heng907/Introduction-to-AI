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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """
        the minimax function:
        1. If depth is 0 or if the game state is a win or a loss (state.isWin() or state.isLose()), the recursion terminates. 
           It returns the evaluation of the current state (self.evaluationFunction(state)),
           and None for the action because no further decisions need to be made.
        2. Fetches all legal actions available to the current agent using state.getLegalActions(agentIndex). 
           If there are no legal actions (which can happen if the game is in a terminal state or if the agent is trapped),
           it returns the evaluation of the state.
        3. Determines which agent will act next. This is computed as (agentIndex + 1) % state.getNumAgents(),
           which cycles through all agents in a loop.
        4. Adjusts the depth for the next recursive call. It decreases by one every time all agents have taken a turn,
           marking a new "level" of the game tree.
        5. Decision Making:
        Maximization for Pacman: If the current agent is Pacman (indicated by agentIndex == 0), it selects the action
        associated with the maximum score using max(results, key=lambda x: x[0]). This is because Pacman aims to maximize his score.
        Minimization for Ghosts: If the current agent is one of the ghosts, it selects the action associated with the minimum score
        using min(results, key=lambda x: x[0]). This models the ghosts' goal to minimize Pacman's score.
        """
        # raise NotImplementedError("To be implemented")
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state), None
            
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth
            
            results = [(minimax(state.getNextState(agentIndex, action), nextDepth, nextAgent)[0], action) for action in actions]
            
            if agentIndex == 0:  # Maximize for Pacman
                return max(results, key=lambda x: x[0])
            else:  # Minimize for ghosts
                return min(results, key=lambda x: x[0])

        # Begin minimax recursion with the current gameState, full search depth, and Pacman (agentIndex 0)
        result = minimax(gameState, self.depth, 0)
        return result[1]  # Return the action that leads to the best outcome

        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """
        1. Checks if the maximum depth is reached or if the game has reached a winning or losing state.
           If so, it returns the evaluation of the current state (no action is returned).
        2. Retrieves all legal actions for the current agent. If no actions are available (terminal state),
           it returns the evaluation of the current state.Determines the next agent and the next depth based on
           the current agent's index.
        3. Initializes value to negative infinity and iterates through each legal action. For each action,
           computes the resulting game state and recursively applies the alphabeta function. Updates value
           if the returned value from recursion is greater (seeking to maximize). Prunes the remaining branches
           if the current value is greater than beta. Updates alpha to the maximum of the current alpha and value.
        4. Similar to the maximizing player but initializes value to positive infinity and seeks to minimize the value.
        5. For Pacman, tracks the best action associated with the maximum value found; for ghosts, tracks the action
           associated with the minimum value found.
        """
        # raise NotImplementedError("To be implemented")
                
        def alphabeta(state, depth, agent, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            legal_actions = state.getLegalActions(agent)
            if not legal_actions:
                return self.evaluationFunction(state), None

            next_agent = (agent + 1) % state.getNumAgents()
            next_depth = depth - 1 if next_agent == 0 else depth

            if agent == 0:  # Pacman, maximizing player
                value = float('-Inf')
                best_action = None
                for action in legal_actions:
                    next_state = state.getNextState(agent, action)
                    next_value, _ = alphabeta(next_state, next_depth, next_agent, alpha, beta)
                    if next_value > value:
                        value = next_value
                        best_action = action
                    if value > beta:
                        return value, action
                    alpha = max(alpha, value)
                return value, best_action

            else:  # Ghosts, minimizing players
                value = float('Inf')
                best_action = None
                for action in legal_actions:
                    next_state = state.getNextState(agent, action)
                    next_value, _ = alphabeta(next_state, next_depth, next_agent, alpha, beta)
                    if next_value < value:
                        value = next_value
                        # best_action = action
                    elif value == next_value:
                        best_action = action
                    if value < alpha:
                        return value, action
                    beta = min(beta, value)
                return value, best_action

        # Start the Alpha-Beta recursion from the root game state with initial alpha and beta values
        _, action = alphabeta(gameState, self.depth, 0, float('-Inf'), float('Inf'))
        return action


        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """
        1. If the recursion reaches the maximum allowed depth or the state is a win or lose situation,
           the function returns the evaluation of that state using self.evaluationFunction(state).
        2. If no actions are available (which can happen in terminal states), it returns the evaluation of the state directly.
        3. Determines which agent will act next using modulo arithmetic. This cycles through agents sequentially,
           resetting to Pacman after all ghosts have taken their turns.
        4. Adjusts the depth for the next recursive call, decreasing only when all agents (including all ghosts) have taken a turn.
        5. Pacman (agent == 0): Since Pacman aims to maximize his score, the function calculates the maximum value
           among all possible actions.It also stores which actions lead to this maximum value. If it's the root call
           (depth == self.depth), it randomly selects from the best actions (ties in the maximum score) to add 
           unpredictability to Pacman's behavior.
        6. Ghosts: As non-deterministic agents, ghosts are modeled to choose actions uniformly at random.
          The function calculates the average of the expectimax values of all actions, representing the expected
          value of any action taken by a ghost given the current state.
        """
        # raise NotImplementedError("To be implemented")
        def expectimax(state, depth, agent):
            # Base case: if depth is 0 or state is a terminal state
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            next_agent = (agent + 1) % state.getNumAgents()
            next_depth = depth - 1 if next_agent == 0 else depth
            actions = state.getLegalActions(agent)

            if not actions:  # Check for no legal actions
                return self.evaluationFunction(state)

            # Generate all next states and values
            values = [expectimax(state.getNextState(agent, action), next_depth, next_agent) for action in actions]

            if agent == 0:  # Pacman's turn, find the maximum value
                max_value = max(values)
                best_actions = [actions[i] for i in range(len(actions)) if values[i] == max_value]
                return max_value if depth != self.depth else random.choice(best_actions)
            else:  # Ghosts' turn, calculate the average value
                avg_value = sum(values) / len(values)
                return avg_value

        # Execute expectimax from the current game state with full search depth and starting from Pacman
        return expectimax(gameState, self.depth, 0)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    1. Retrieves Pacman's current position from the game state, which is used to calculate distances to ghosts and food.
    2. Fetches the states of all ghosts, including their positions and whether they are in a scared state (scaredTimer).
       Calculates the Manhattan distances from Pacman to each ghost.
    3. If the total scared time of all ghosts is more than 1, ghosts are vulnerable and Pacman can chase them for points.
       The closer the ghost, the higher the reward if Pacman is on the same tile as a ghost (min_ghost_distance == 0),
       a large score boost (600 points) is added.Otherwise, the score increment is inversely proportional to the distance  
       to the closest ghost; if a ghost is on the same tile as Pacman, it significantly decreases the score (penalty of 100 points).
       if a ghost is very close (distance less than 5), there's a smaller penalty inversely proportional to the distance
       to deter Pacman from getting too close.
    4. Retrieves all food positions as a list and calculates the Manhattan distance from Pacman to each piece of food.
       The score is penalized based on the number of food pieces remaining (-5 points per piece) to encourage Pacman to eat 
       food and reduce this penalty. Additionally, the closest piece of food provides a positive score boost, making nearer
       food more attractive.
       This is calculated as 10 / min_food_distance + 10, giving a significant bonus if food is very close.
    5. Retrieves all capsule positions and each capsule left in the game imposes a penalty of 100 points, encouraging
       Pacman to collect them to reduce the penalty.
    """
    # raise NotImplementedError("To be implemented")
    pac_pos = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    ghost_pos = currentGameState.getGhostPositions()

    # Get the scared times and ghost distances from Pacman
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]
    ghost_distances = [util.manhattanDistance(pac_pos, ghost_position) for ghost_position in ghost_pos]

    # Calculate ghost related scores
    ghost_score = 0
    total_scared_time = sum(scared_times)
    min_ghost_distance = min(ghost_distances) if ghost_distances else float('inf')

    if total_scared_time > 1:
        if min_ghost_distance == 0:
            ghost_score += 600
        else:
            ghost_score += 300 / min_ghost_distance
    else:
        if min_ghost_distance == 0:
            ghost_score -= 100
        elif min_ghost_distance < 5:
            ghost_score -= 20 / min_ghost_distance

    # Calculate food related scores
    food = currentGameState.getFood().asList()
    food_distances = [util.manhattanDistance(pac_pos, food_pos) for food_pos in food]
    food_score = -5 * len(food_distances)
    if food_distances:
        min_food_distance = min(food_distances)
        food_score += 10 / min_food_distance + 10

    # Calculate capsules related scores
    capsules = currentGameState.getCapsules()
    capsules_score = -100 * len(capsules)

    # Calculate the total score by adding all component scores to the game state's score
    return ghost_score + food_score + capsules_score + currentGameState.getScore()


    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction

# def getLegalActionsNoStop(index, gameState):
#     possibleActions = gameState.getLegalActions(index)
#     if Directions.STOP in possibleActions:
#         possibleActions.remove(Directions.STOP)
#     return possibleActions