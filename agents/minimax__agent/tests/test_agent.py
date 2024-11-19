# FILE: minimax_agent/tests/test_agent.py
import unittest
from agent_minimax.agent_code import MinimaxAgent, ConnectFourGame

class TestMinimaxAgent(unittest.TestCase):
    def setUp(self):
        self.game = ConnectFourGame()
        self.agent = MinimaxAgent(depth=2)

    def test_select_move(self):
        move = self.agent.select_move(self.game)
        self.assertIn(move, range(7))  # Assuming 7 columns (0-6)

    def test_win_detection(self):
        # Simulate a simple win scenario
        for i in range(4):
            row = self.game.get_next_open_row(0)
            self.game.drop_piece(row, 0, 1)
        self.assertTrue(self.game.winning_move(PLAYER_PIECE))

if __name__ == '__main__':
    unittest.main()