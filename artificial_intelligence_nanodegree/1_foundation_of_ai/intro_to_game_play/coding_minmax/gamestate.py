from copy import deepcopy

# Define board dimensions
xlim = 3
ylim = 2


class GameState:
    """Attributes:
    _board: list(list)
         Represents the board with a 2d array _board[x][y] where
        open spaces are 0 and closed spaces are 1.
    _parity: bool
        Keep track of active player initiative (which player has
        control to move) where 0 indicates that player one has
        initiative nd 1 indicates player 2.
    _player_locations: list(tuple)
        Keep track of the current location of each player on the
        board where position is encoded by the board indices of
        their last move, e.g., [(0, 0), (1, 0)] means player 1 is
        at (0, 0) and player 2 is at (1, 0).
     """

    def __init__(self):
        self._board = [[0] * ylim for _ in range(xlim)]
        self._board[-1][-1] = 1  # block lower right corner
        self._parity = 0
        self._player_locations = [None, None]

    def forecast_move(self, move):
        """ Return a new board object with the specified move
        applied to the current game state.

        Parameters
        ----------
        move: tuple
            The target position for the active player's next move
        """
        # Check if the move is legal
        if move not in self.get_legal_moves():
            raise RuntimeError("Attempted forecast of illegal move")

        # Copy the board
        newBoard = deepcopy(self)

        # Update the board
        newBoard._board[move[0]][move[1]] = 1

        # Update current player position
        newBoard._player_locations[self._parity] = move

        # Use XOR to switch player
        newBoard._parity ^= 1

        return newBoard

    def get_legal_moves(self):
        """ Return a list of all legal moves available to the
        active player.  Each player should get a list of all
        empty spaces on the board on their first move, and
        otherwise they should get a list of all open spaces
        in a straight line along any row, column or diagonal
        from their current position. (Players CANNOT move
        through obstacles or blocked squares.) Moves should
        be a pair of integers in (column, row) order specifying
        the zero-indexed coordinates on the board.
        """
        # Get current position
        current_loc = self._player_locations[self._parity]
        if current_loc is None:
            return self._get_blank_spaces()

        # Create posible moving increments
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                      (-1, 0), (-1, -1), (0, -1), (1, -1)]

        # Loop through the moving increments to find all possible
        # moves.
        legal_moves = []
        for dx, dy in directions:
            # new_x, new_y = self._new_loc(current_loc, d)
            _x, _y = current_loc
            # Check whether the new location is on the board
            while (0 <= _x + dx < xlim and 0 <= _y + dy < ylim):
                _x, _y = _x + dx, _y + dy
                # If the position is occupied, then break
                if self._board[_x][_y]:
                    break
                # If the move is legal, then we appended
                legal_moves.append((_x, _y))

        return legal_moves

    def _get_blank_spaces(self):
        """ Obtains all currently unoccupied squares.
        """
        return [(x, y)
                for x in range(xlim)
                for y in range(ylim)
                if self._board[x][y] == 0]
