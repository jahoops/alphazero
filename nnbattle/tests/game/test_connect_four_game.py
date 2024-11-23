import unittest
import numpy as np
import logging

from nnbattle.game.connect_four_game import ConnectFourGame, EMPTY, RED_PIECE, YEL_PIECE, COLUMN_COUNT, ROW_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestConnectFourGame(unittest.TestCase):
    def setUp(self):
        self.game = ConnectFourGame()
        # Enable turn enforcement by removing the line that disables it
        # self.game.enforce_turns = False  # Removed to enforce alternating turns

    def assertWithBoardState(self, assertion, message=None):
        """Helper method to show board state on assertion failure"""
        try:
            assertion()
        except AssertionError as e:
            board_state = f"\nCurrent board state:\n{self.game.board_to_string()}"
            raise AssertionError(f"{str(e)}{board_state}")

    def test_initial_state(self):
        self.assertWithBoardState(
            lambda: self.assertTrue(np.array_equal(self.game.board, np.zeros((6,7), dtype=np.int8)))
        )
        self.assertWithBoardState(
            lambda: self.assertIsNone(self.game.last_piece)
        )

    def test_make_move_valid(self):
        # First move can be either piece
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.make_move(0, RED_PIECE))
        )
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.board[0][0], RED_PIECE)
        )
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.last_piece, RED_PIECE)
        )

        # Next move must be the other piece
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.make_move(1, YEL_PIECE))
        )
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.board[0][1], YEL_PIECE)
        )
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.last_piece, YEL_PIECE)
        )

    def test_make_move_invalid_turn(self):
        self.game.enforce_turns = True
        self.game.make_move(0, RED_PIECE)
        self.assertWithBoardState(
            lambda: self.assertFalse(self.game.make_move(1, RED_PIECE))  # Same piece can't move twice
        )

    def test_make_move_invalid_column(self):
        self.assertWithBoardState(
            lambda: self.assertFalse(self.game.make_move(-1, RED_PIECE))
        )
        self.assertWithBoardState(
            lambda: self.assertFalse(self.game.make_move(COLUMN_COUNT, RED_PIECE))
        )

    def test_make_move_full_column(self):
        for _ in range(ROW_COUNT):
            self.assertWithBoardState(
                lambda: self.game.make_move(0, RED_PIECE if _ % 2 == 0 else YEL_PIECE)
            )
        self.assertWithBoardState(
            lambda: self.assertFalse(self.game.make_move(0, RED_PIECE))
        )

    def test_check_win_horizontal(self):
        # Disable turn enforcement for setting up win condition
        self.game.enforce_turns = False
        # Make four moves in a row for RED_PIECE
        for c in range(4):
            self.assertWithBoardState(
                lambda: self.assertTrue(self.game.make_move(c, RED_PIECE), 
                          f"Move {c} failed\n{self.game.board_to_string()}")
            )
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.check_win(RED_PIECE))
        )

    def test_check_win_vertical(self):
        # Disable turn enforcement for setting up win condition
        self.game.enforce_turns = False
        # Make four moves in the same column for RED_PIECE
        for _ in range(4):
            self.assertWithBoardState(
                lambda: self.assertTrue(self.game.make_move(0, RED_PIECE),
                          f"Vertical move failed\n{self.game.board_to_string()}")
            )
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.check_win(RED_PIECE))
        )

    def test_check_win_positive_diagonal(self):
        # Setup a diagonal win pattern
        moves = [(0, RED_PIECE), (1, YEL_PIECE), 
                (1, RED_PIECE), (2, YEL_PIECE),
                (2, RED_PIECE), (3, YEL_PIECE),
                (2, RED_PIECE), (3, YEL_PIECE),
                (3, RED_PIECE), (0, YEL_PIECE),
                (3, RED_PIECE)]
        for col, piece in moves:
            self.assertWithBoardState(
                lambda: self.game.make_move(col, piece)
            )
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.check_win(RED_PIECE))
        )

    def test_check_win_negative_diagonal(self):
        # Setup a negative diagonal win pattern
        moves = [(3, RED_PIECE), (2, YEL_PIECE),
                (2, RED_PIECE), (1, YEL_PIECE),
                (1, RED_PIECE), (0, YEL_PIECE),
                (1, RED_PIECE), (0, YEL_PIECE),
                (0, RED_PIECE), (3, YEL_PIECE),
                (0, RED_PIECE)]
        for col, piece in moves:
            self.assertWithBoardState(
                lambda: self.game.make_move(col, piece)
            )
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.check_win(RED_PIECE))
        )

    def test_is_board_full(self):
        # Fill board alternating pieces
        for c in range(COLUMN_COUNT):
            for _ in range(ROW_COUNT):
                self.assertWithBoardState(
                    lambda: self.game.make_move(c, RED_PIECE if _ % 2 == 0 else YEL_PIECE)
                )
        self.assertWithBoardState(
            lambda: self.assertTrue(self.game.is_board_full())
        )

    def test_get_game_state(self):
        # Test initial state
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.get_game_state(), "ONGOING")
        )
        
        # Test horizontal RED win - place pieces in columns 0,1,2,3
        self.game.enforce_turns = False  # Disable turn enforcement
        for column in range(4):  # More explicit naming
            self.assertWithBoardState(
                lambda: self.assertTrue(self.game.make_move(column, RED_PIECE), 
                          f"Failed to make move in column {column}")
            )
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.get_game_state(), "RED_WINS")
        )
    
        # Test vertical YEL win in new game - place 4 pieces in column 0
        self.game = ConnectFourGame()
        self.game.enforce_turns = False
        for row in range(4):  # More explicit naming
            self.assertWithBoardState(
                lambda: self.assertTrue(self.game.make_move(0, YEL_PIECE),
                          f"Failed to make move in row {row}")
            )
        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.get_game_state(), "YEL_WINS")
        )

        # Test draw state
        self.game = ConnectFourGame()
        self.game.enforce_turns = False

        # Define the draw board
        draw_board = np.array([
            [RED_PIECE, RED_PIECE, RED_PIECE, YEL_PIECE, YEL_PIECE, YEL_PIECE, RED_PIECE],
            [YEL_PIECE, YEL_PIECE, YEL_PIECE, RED_PIECE, RED_PIECE, RED_PIECE, YEL_PIECE],
            [RED_PIECE, RED_PIECE, RED_PIECE, YEL_PIECE, YEL_PIECE, YEL_PIECE, RED_PIECE],
            [YEL_PIECE, YEL_PIECE, YEL_PIECE, RED_PIECE, RED_PIECE, RED_PIECE, YEL_PIECE],
            [RED_PIECE, RED_PIECE, RED_PIECE, YEL_PIECE, YEL_PIECE, YEL_PIECE, RED_PIECE],
            [YEL_PIECE, YEL_PIECE, YEL_PIECE, RED_PIECE, RED_PIECE, RED_PIECE, YEL_PIECE],
        ], dtype=np.int8)
        
        # Set the board to the draw board
        self.game.board = draw_board

        self.assertWithBoardState(
            lambda: self.assertEqual(self.game.get_game_state(), "DRAW")
        )

    def test_new_game(self):
        self.assertWithBoardState(
            lambda: self.game.make_move(0, RED_PIECE)
        )
        new_game = self.game.new_game()
        self.assertWithBoardState(
            lambda: self.assertTrue(np.array_equal(new_game.board, np.zeros((6,7), dtype=np.int8)))
        )
        self.assertWithBoardState(
            lambda: self.assertIsNone(new_game.last_piece)
        )

    def test_make_move_turn_enforcement(self):
        """Test that turn enforcement works when enabled"""
        game = ConnectFourGame()  # New game with enforcement enabled
        game.enforce_turns = True
        
        # First move should work
        self.assertWithBoardState(
            lambda: self.assertTrue(game.make_move(0, RED_PIECE))
        )
        # Same piece shouldn't be able to move again
        self.assertWithBoardState(
            lambda: self.assertFalse(game.make_move(1, RED_PIECE))
        )
        # Other piece should be able to move
        self.assertWithBoardState(
            lambda: self.assertTrue(game.make_move(1, YEL_PIECE))
        )

if __name__ == '__main__':
    unittest.main()