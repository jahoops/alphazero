# game/tests/test_connect_four_game.py

import unittest
import numpy as np
import logging

from nnbattle.game.connect_four_game import ConnectFourGame, EMPTY, PLAYER_PIECE, AI_PIECE, COLUMN_COUNT, ROW_COUNT

# Configure logging for the test module
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO
logger = logging.getLogger(__name__)


class TestConnectFourGame(unittest.TestCase):
    def setUp(self):
        self.game = ConnectFourGame()

    def board_to_string(self, board):
        """
        Converts the board state to a string representation using 'X' for PLAYER_PIECE,
        'O' for AI_PIECE, and '.' for EMPTY.
        
        :param board: NumPy array representing the board state.
        :return: String representation of the board.
        """
        symbol_mapping = {PLAYER_PIECE: 'X', AI_PIECE: 'O', EMPTY: '.'}
        rows = []
        for row in board:
            rows.append(' '.join([symbol_mapping[cell] for cell in row]))
        return '\n'.join(rows)

    def test_initial_state(self):
        expected_board = np.zeros((6,7), dtype=np.int8)
        self.assertTrue(np.array_equal(self.game.board, expected_board))
        self.assertEqual(self.game.current_player, PLAYER_PIECE)

    def test_make_move_valid(self):
        self.game.make_move(0)
        self.assertEqual(self.game.board[0][0], PLAYER_PIECE)
        self.assertEqual(self.game.current_player, AI_PIECE)

    def test_make_move_invalid(self):
        for _ in range(6):
            self.game.make_move(0)
        with self.assertLogs(level='ERROR') as log:
            self.game.make_move(0)
            self.assertIn("Invalid move attempted: Column 0 is full.", log.output[0])

    def test_is_valid_location(self):
        self.assertTrue(self.game.is_valid_location(0))
        for _ in range(6):
            self.game.make_move(0)
        self.assertFalse(self.game.is_valid_location(0))

    def test_get_next_open_row(self):
        self.assertEqual(self.game.get_next_open_row(0), 0)
        self.game.make_move(0)
        self.assertEqual(self.game.get_next_open_row(0), 1)

    def test_drop_piece(self):
        self.game.drop_piece(0, 0, PLAYER_PIECE)
        self.assertEqual(self.game.board[0][0], PLAYER_PIECE)

    def test_check_win_horizontal(self):
        for c in range(4):
            self.game.make_move(c)
            self.game.make_move(c)
        self.assertTrue(self.game.check_win(PLAYER_PIECE))

    def test_check_win_vertical(self):
        for _ in range(4):
            self.game.make_move(0)
            self.game.make_move(1)
        self.assertTrue(self.game.check_win(PLAYER_PIECE))

    def test_check_win_positive_diagonal(self):
        moves = [0,1,1,2,2,3,2,3,3,4,3]
        for move in moves:
            self.game.make_move(move)
        self.assertTrue(self.game.check_win(PLAYER_PIECE))

    def test_check_win_negative_diagonal(self):
        moves = [3,2,2,1,1,0,1,0,0,0]
        for move in moves:
            self.game.make_move(move)
        self.assertTrue(self.game.check_win(PLAYER_PIECE))

    def test_is_board_full(self):
        for c in range(7):
            for _ in range(6):
                self.game.make_move(c)
        self.assertTrue(self.game.is_board_full())
        self.assertFalse(self.game.is_terminal())  # Ensure no winner results in terminal

    def test_get_winner_player(self):
        for c in range(4):
            self.game.make_move(c)
            self.game.make_move(c)
        self.assertEqual(self.game.get_winner(), PLAYER_PIECE)

    def test_get_winner_ai(self):
        for _ in range(3):
            self.game.make_move(0)
            self.game.make_move(1)
        self.game.make_move(0)
        self.assertEqual(self.game.get_winner(), PLAYER_PIECE)

    def test_get_winner_none(self):
        # Fill the board without any player winning
        for c in range(7):
            for _ in range(6):
                self.game.make_move(c)
        self.assertEqual(self.game.get_winner(), 0)

    def test_get_reward_player_win(self):
        for c in range(4):
            self.game.make_move(c)
            self.game.make_move(c)
        self.assertEqual(self.game.get_reward(), 1)

    def test_get_reward_ai_win(self):
        for _ in range(4):
            self.game.make_move(0)
            self.game.make_move(1)
        self.assertEqual(self.game.get_reward(), -1)

    def test_get_reward_draw(self):
        # Fill the board without any winner
        for c in range(7):
            for _ in range(6):
                self.game.make_move(c)
        self.assertEqual(self.game.get_reward(), 0)

    def test_preprocess_board(self):
        self.game.make_move(0)  # Player
        preprocessed = self.game.get_state()
        expected = np.zeros((2,6,7))
        expected[0,0,0] = 1  # Current player
        expected[1] = 0  # Opponent
        self.assertTrue(np.array_equal(preprocessed, expected))

    def test_is_terminal_win(self):
        """Simulate a vertical win for PLAYER_PIECE correctly by making four consecutive moves in the same column."""
        move = 0
        for _ in range(4):
            self.assertTrue(self.game.make_move(move), "Move should be successful.")
            # Ensure all moves are made by PLAYER_PIECE by not switching players
            # This is necessary because make_move() switches the player after each move
            self.game.current_player = PLAYER_PIECE
        # Log the board state for debugging
        logger.info("Board state after setting up a vertical win for PLAYER_PIECE:")
        logger.info(f"\n{self.board_to_string(self.game.board)}")
        self.assertTrue(self.game.is_terminal(), "Game should be terminal after a win.")
        self.assertEqual(self.game.get_winner(), PLAYER_PIECE, "PLAYER_PIECE should be the winner.")

    def test_is_terminal_draw(self):
        """
        Tests that a fully populated board without any winning sequences
        is correctly identified as a terminal draw.
        """
        # Define a board state that is full but has no four-in-a-row for any player
        # This board is manually crafted to avoid any horizontal, vertical, or diagonal wins
        draw_board = np.array([
            [PLAYER_PIECE, PLAYER_PIECE, PLAYER_PIECE, AI_PIECE, AI_PIECE, AI_PIECE, PLAYER_PIECE],
            [AI_PIECE, AI_PIECE, AI_PIECE, PLAYER_PIECE, PLAYER_PIECE, PLAYER_PIECE, AI_PIECE],
            [PLAYER_PIECE, PLAYER_PIECE, PLAYER_PIECE, AI_PIECE, AI_PIECE, AI_PIECE, PLAYER_PIECE],
            [AI_PIECE, AI_PIECE, AI_PIECE, PLAYER_PIECE, PLAYER_PIECE, PLAYER_PIECE, AI_PIECE],
            [PLAYER_PIECE, PLAYER_PIECE, PLAYER_PIECE, AI_PIECE, AI_PIECE, AI_PIECE, PLAYER_PIECE],
            [AI_PIECE, AI_PIECE, AI_PIECE, PLAYER_PIECE, PLAYER_PIECE, PLAYER_PIECE, AI_PIECE],
        ], dtype=np.int8)

        # Assign the predefined draw board to the game
        self.game.board = draw_board

        # Log the board state for debugging
        logger.info("Board state for draw scenario:")
        logger.info(f"\n{self.board_to_string(self.game.board)}")

        # Assert that the game is terminal
        self.assertTrue(self.game.is_terminal(), "Game should be terminal after a draw.")

        # Assert that there is no winner
        winner = self.game.get_winner()
        self.assertEqual(winner, EMPTY, "There should be no winner in a draw.")

        # Additional assertions to ensure no player has a winning condition
        self.assertFalse(self.game.check_win(AI_PIECE), "AI_PIECE should not have a winning condition in a draw.")
        self.assertFalse(self.game.check_win(PLAYER_PIECE), "PLAYER_PIECE should not have a winning condition in a draw.")

    def test_score_position_center(self):
        # Ensure AI_PIECE controls the center
        center_col = COLUMN_COUNT // 2
        # PLAYER_PIECE makes a move in the center
        self.game.make_move(center_col)
        # AI_PIECE makes a move in the center
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

    def test_board_type_after_move(self):
        """Ensure that self.board remains a NumPy array after making a move."""
        move = 3
        self.game.make_move(move)
        self.assertIsInstance(self.game.board, np.ndarray, "self.board should remain a NumPy array after a move.")

    def test_is_valid_location_assertion(self):
        """Test that is_valid_location raises an assertion error when self.board is not a NumPy array."""
        with self.assertRaises(AssertionError):
            self.game.board = "not a numpy array"  # Intentionally set board to an incorrect type
            self.game.is_valid_location(3)

if __name__ == '__main__':
    unittest.main()