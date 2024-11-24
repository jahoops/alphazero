# game/connect_four_game.py

import copy
import logging
import numpy as np
from nnbattle.constants import RED_TEAM, YEL_TEAM, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InvalidMoveError(Exception):
    """Exception raised when an invalid move is made."""
    pass

class InvalidTurnError(Exception):
    """Exception raised when a player tries to move out of turn."""
    pass

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.last_piece = None
        self.enforce_turns = True  # Always enforce turns

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
        """Make a move for the given piece in the specified column."""
        if piece not in [RED_TEAM, YEL_TEAM]:
            logger.error(f"Invalid piece: {piece}. Must be {RED_TEAM} or {YEL_TEAM}.")
            raise InvalidMoveError(f"Invalid piece: {piece}. Must be {RED_TEAM} or {YEL_TEAM}.")

        if self.enforce_turns:
            if self.last_piece is not None and piece == self.last_piece:
                logger.error(f"Invalid turn: Piece {piece} cannot move twice in a row.")
                raise InvalidTurnError(f"Invalid turn: Piece {piece} cannot move twice in a row.")

        if not self.is_valid_move(column):
            logger.error(f"Invalid move: Column {column} is full or out of bounds.")
            raise InvalidMoveError(f"Invalid move: Column {column} is full or out of bounds.")

        row = self.get_next_open_row(column)
        self.board[row][column] = piece
        self.last_piece = piece
        logger.debug(f"Piece {piece} placed in column {column}, row {row}.")
        return True

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