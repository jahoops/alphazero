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
        """Simulate a vertical win for PLAYER_PIECE correctly by making four consecutive moves in the same column."""
        move = 0
        for _ in range(4):
            self.assertTrue(self.game.make_move(move), "Move should be successful.")
            # Ensure all moves are made by PLAYER_PIECE by not switching players
            # This is necessary because make_move() switches the player after each move
            self.game.current_player = PLAYER_PIECE
        # Log the board state for debugging
        logger.debug("Board state after setting up a vertical win for PLAYER_PIECE:")
        logger.debug(f"\n{self.board_to_string(self.game.board)}")
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
        # logger.debug("Board state for draw scenario:")  # Consider removing or commenting out
        # logger.debug(f"\n{self.board_to_string(self.game.board)}")  # Consider removing or commenting out

        # Assert that the game is terminal
        self.assertTrue(self.game.is_terminal(), "Game should be terminal after a draw.")

        # Assert that there is no winner
        winner = self.game.get_winner()
        # self.assertEqual(winner, EMPTY, "There should be no winner in a draw.")  # ...existing code...

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