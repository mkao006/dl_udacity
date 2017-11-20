def terminal_test(gameState):
    """ Return True if the game is over for the active player
    and False otherwise.
    """
    # Boolean of empty list gives False
    return not bool(gameState.get_legal_moves())


def min_value(gameState):
    """ Return the value for a win (+1) if the game is over,
    otherwise return the minimum value over all legal child
    nodes.
    """

    if terminal_test(gameState):
        return 1

    utility = float('inf')
    for l in gameState.get_legal_moves():
        utility = min(utility, max_value(gameState.forecast_move(l)))
    return utility


def max_value(gameState):
    """ Return the value for a loss (-1) if the game is over,
    otherwise return the maximum value over all legal child
    nodes.
    """
    if terminal_test(gameState):
        return -1

    utility = float('-inf')
    for l in gameState.get_legal_moves():
        utility = max(utility, min_value(gameState.forecast_move(l)))

    return utility
