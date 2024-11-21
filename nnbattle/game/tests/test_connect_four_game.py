# game/tests/test_connect_four_game.py

import unittest

from nnbattle.game.connect_four_game import ConnectFourGame, EMPTY, PLAYER_PIECE, AI_PIECE


class TestConnectFourGame(unittest.TestCase):
    def setUp(self):
        self.game = ConnectFourGame()

    def test_initial_state(self):
        self.assertTrue((self.game.board == EMPTY).all(), "Initial board should be all zeros.")
        self.assertEqual(self.game.current_player, PLAYER_PIECE, "Player should start first.")

    def test_make_move_valid(self):
        move = 3
        success = self.game.make_move(move)
        self.assertTrue(success, "Move should be successful.")
        self.assertEqual(self.game.board[5][move], PLAYER_PIECE, "Piece should be placed at the bottom of the column.")
        self.assertEqual(self.game.current_player, AI_PIECE, "Current player should switch to AI.")

    def test_make_move_invalid(self):
        move = 0
        # Fill the first column
        for _ in range(6):
            self.assertTrue(self.game.make_move(move), "Moves should be successful until the column is full.")
        # Attempt to make a move in the full column
        success = self.game.make_move(move)
        self.assertFalse(success, "Move should fail when column is full.")
        self.assertEqual(self.game.current_player, PLAYER_PIECE, "Current player should not switch after invalid move.")

    def test_is_terminal_win(self):
        # Simulate a vertical win for PLAYER_PIECE
        move = 0
        for _ in range(4):
            self.game.make_move(move)
            move += 1  # Alternate moves to prevent opponent from blocking
        self.assertTrue(self.game.is_terminal(), "Game should be terminal after a win.")
        self.assertEqual(self.game.get_winner(), PLAYER_PIECE, "PLAYER_PIECE should be the winner.")

    def test_is_terminal_draw(self):
        # Fill the board without any winning moves
        for col in range(7):
            for _ in range(6):
                self.game.make_move(col)
        self.assertTrue(self.game.is_terminal(), "Game should be terminal after a draw.")
        self.assertEqual(self.game.get_winner(), EMPTY, "There should be no winner in a draw.")

    def test_score_position_center(self):
        # Place a piece in the center
        center_col = COLUMN_COUNT // 2
        self.game.make_move(center_col)
        score = self.game.score_position(AI_PIECE)
        self.assertGreater(score, 0, "Score should increase when AI controls the center.")

    def test_score_position_win(self):
        # Simulate a horizontal win for AI_PIECE
        for col in range(4):
            self.game.make_move(col)
            self.game.make_move(col)  # Switch to AI and back
        score = self.game.score_position(AI_PIECE)
        self.assertGreater(score, 0, "Score should be high when AI has potential winning moves.")

if __name__ == '__main__':
    unittest.main()