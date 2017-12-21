# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""In search.py, you will implement generic search algorithms which
are called by Pacman agents (in searchAgents.py).

"""

import util


class SearchProblem:
    """This class outlines the structure of a search problem, but doesn't
    implement any of the methods (in object-oriented terminology: an
    abstract class).

    You do not need to change anything in this class, ever.

    """

    def getStartState(self):
        """Returns the start state for the search problem

        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """state: Search state

         Returns True if and only if the state is a valid goal state

        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """state: Search state

         For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental cost
         of expanding to that successor

        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """actions: A list of actions to take

         This method returns the total cost of a particular sequence
         of actions.  The sequence must be composed of legal moves

        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this
    for tinyMaze

    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def transition_model(current_state, action):
    """ Function to update the state and store the historical actions.
    """

    new_state = current_state.copy()
    new_state['current_location'] = action[0]
    new_state['actions'] = current_state['actions'] + [action[1]]
    new_state['costs'] += action[2]
    return new_state


def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that
    reaches the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    """

    # Initialisate state
    initial_state = {'current_location': problem.getStartState(),
                     'actions': [],
                     'costs': 0}

    # Initialise Frontier
    frontier = util.Stack()
    frontier.push(initial_state)
    # Intialise Explored - (empty hashed dictionary)
    explored = {}

    # Start loop
    while not frontier.isEmpty():
        # pop a frontier to explored
        current_state = frontier.pop()

        # if the node is a solution, then break from loop
        if problem.isGoalState(current_state['current_location']):
            break

        # add the location tuple to the explored
        explored[hash(current_state['current_location'])] = True

        # For each children of the node, create a new node with
        # history and added to frontier.
        for child in problem.getSuccessors(current_state['current_location']):
            if hash(child[0]) not in explored:
                frontier.push(transition_model(current_state, child))

    return current_state['actions']


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    # Initialisate state
    initial_state = {'current_location': problem.getStartState(),
                     'actions': [],
                     'costs': 0}

    # Initialise Frontier
    frontier = util.Queue()
    frontier.push(initial_state)
    # Intialise Explored - (empty hashed dictionary)
    explored = {}

    # Start loop
    while not frontier.isEmpty():
        # pop a frontier to explored
        current_state = frontier.pop()

        # if the node is a solution, then break from loop
        if problem.isGoalState(current_state['current_location']):
            break

        # add the location tuple to the explored
        explored[hash(current_state['current_location'])] = True

        # For each children of the node, create a new node with
        # history and added to frontier.
        for child in problem.getSuccessors(current_state['current_location']):
            if hash(child[0]) not in explored:
                frontier.push(transition_model(current_state, child))

    return current_state['actions']


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    # Initialisate state
    initial_state = {'current_location': problem.getStartState(),
                     'actions': [],
                     'costs': 0}

    # Initialise Frontier
    frontier = util.PriorityQueue()
    frontier.push(initial_state, initial_state['costs'])
    # Intialise Explored - (empty hashed dictionary)
    explored = {}

    # Start loop
    while not frontier.isEmpty():
        # pop a frontier to explored
        current_state = frontier.pop()

        # if the node is a solution, then break from loop
        if problem.isGoalState(current_state['current_location']):
            break

        # add the location tuple to the explored
        explored[hash(current_state['current_location'])] = True

        # For each children of the node, create a new node with
        # history and added to frontier.
        for child in problem.getSuccessors(current_state['current_location']):
            if hash(child[0]) not in explored:
                new_state = transition_model(current_state, child)
                frontier.push(new_state, new_state['costs'])

    return current_state['actions']


def nullHeuristic(state, problem=None):
    """A heuristic function estimates the cost from the current state to
    the nearest goal in the provided SearchProblem.  This heuristic is
    trivial.

    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    # Initialisate state
    initial_state = {'current_location': problem.getStartState(),
                     'actions': [],
                     'costs': 0}

    # Initialise Frontier
    frontier = util.PriorityQueue()
    frontier.push(initial_state,
                  heuristic(initial_state['current_location'], problem))
    # Intialise Explored - (empty hashed dictionary)
    explored = {}

    # Start loop
    while not frontier.isEmpty():
        # pop a frontier to explored
        current_state = frontier.pop()

        # if the node is a solution, then break from loop
        if problem.isGoalState(current_state['current_location']):
            break

        # add the location tuple to the explored
        explored[hash(current_state['current_location'])] = True

        # For each children of the node, create a new node with
        # history and added to frontier.
        for child in problem.getSuccessors(current_state['current_location']):
            if hash(child[0]) not in explored:
                new_state = transition_model(current_state, child)
                frontier.push(new_state,
                              heuristic(new_state['current_location'], problem))

    return current_state['actions']


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
