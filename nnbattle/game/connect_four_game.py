# game/connect_four_game.py

import copy
import logging
import numpy as np
from nnbattle.constants import RED_TEAM, YEL_TEAM, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.last_piece = None  # Track last piece played
        self.enforce_turns = True  # Add flag to control turn enforcement

    def new_game(self):
        """Creates and returns a new game instance."""
        new_game = ConnectFourGame()
        new_game.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        new_game.last_piece = None
        return new_game

    def reset(self):
        """Resets the game to initial state."""
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.last_piece = None
        return self.board.copy()

    def make_move(self, column, piece):
        """Make a move for the given piece in the specified column.
        Returns True if move was valid and made, False otherwise."""
        if not (piece in [RED_TEAM, YEL_TEAM]):
            logger.error(f"Invalid piece: {piece}. Must be {RED_TEAM} or {YEL_TEAM}")
            return False
            
        # Check turn order only if enforcement is enabled
        if self.enforce_turns and self.last_piece is not None and piece == self.last_piece:
            logger.error(f"Invalid turn: Piece {piece} cannot move twice in a row")
            return False

        if self.is_valid_move(column):
            row = self.get_next_open_row(column)
            self.board[row][column] = piece
            self.last_piece = piece
            return True
            
        logger.error(f"Invalid move: Column {column} is full")
        return False

    def is_valid_move(self, column):
        """Check if a move is valid."""
        return (0 <= column < COLUMN_COUNT and 
                self.board[ROW_COUNT-1][column] == EMPTY)

    def get_next_open_row(self, column):
        """Get the next available row in the given column."""
        for row in range(ROW_COUNT):
            if self.board[row][column] == EMPTY:
                return row
        return None

    def check_win(self, piece):
        """Check if the given piece has won."""
        # Check horizontal
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r][c+i] == piece for i in range(4)):
                    return True

        # Check vertical
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT):
                if all(self.board[r+i][c] == piece for i in range(4)):
                    return True

        # Check positive diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r+i][c+i] == piece for i in range(4)):
                    return True

        # Check negative diagonal
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r-i][c+i] == piece for i in range(4)):
                    return True

        return False

    def is_board_full(self):
        """Check if the board is full."""
        return not any(self.board[ROW_COUNT-1][c] == EMPTY for c in range(COLUMN_COUNT))

    def get_game_state(self):
        """Return the current game state."""
        if self.check_win(RED_TEAM):
            return RED_TEAM
        elif self.check_win(YEL_TEAM):
            return YEL_TEAM
        elif self.is_board_full():
            return "Draw"
        return "ONGOING"

    def get_valid_moves(self):
        """Return list of valid column moves."""
        return [col for col in range(COLUMN_COUNT) if self.is_valid_move(col)]

    def get_board(self):
        """Return copy of current board state."""
        return self.board.copy()

    def board_to_string(self):
        """Return string representation of board."""
        mapping = {RED_TEAM: 'X', YEL_TEAM: 'O', EMPTY: '.'}
        return '\n'.join(' '.join(mapping[cell] for cell in row) for row in self.board)