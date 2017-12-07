"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import timeit
import isolation
import game_agent as ga

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(ga)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)
        self.time_limit = 150

    def _time_millis(self):
        return 1000 * timeit.default_timer()

    def _time_left(self, move_start):
        def time_left():
            return self.time_limit - (self._time_millis() - move_start)
        return time_left

    def test_move_output(self):
        player = ga.MinimaxPlayer(search_depth=3, score_fn=ga.custom_score)
        move_start = self._time_millis()
        move = player.get_move(self.game, self._time_left(move_start))
        print(move)
        self.assertIsInstance(move, tuple)

    def test_minimax_output(self):
        pass


if __name__ == '__main__':
    unittest.main()
