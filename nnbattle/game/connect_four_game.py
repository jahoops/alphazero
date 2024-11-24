# game/connect_four_game.py

import copy
import logging
import numpy as np
from nnbattle.constants import RED_TEAM, YEL_TEAM, EMPTY, ROW_COUNT, COLUMN_COUNT

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
        self.last_team = None
        self.enforce_turns = True  # Always enforce turns

    def new_game(self):
        """Creates and returns a new game instance."""
        new_game = ConnectFourGame()
        new_game.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        new_game.last_team = None
        return new_game
    def reset(self):
        """Resets the game to initial state."""
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.last_team = None
        return self.board.copy()

    def make_move(self, column, team):
        """Make a move for the given team in the specified column."""
        if team not in [RED_TEAM, YEL_TEAM]:
            logger.error(f"Invalid team: {team}. Must be {RED_TEAM} or {YEL_TEAM}.")
            raise InvalidMoveError(f"Invalid team: {team}. Must be {RED_TEAM} or {YEL_TEAM}.")

        if self.enforce_turns:
            if self.last_team is not None and team == self.last_team:
                logger.error(f"Invalid turn: team {team} cannot move twice in a row.")
                raise InvalidTurnError(f"Invalid turn: team {team} cannot move twice in a row.")

        if not self.is_valid_move(column):
            logger.error(f"Invalid move: Column {column} is full or out of bounds.")
            raise InvalidMoveError(f"Invalid move: Column {column} is full or out of bounds.")

        row = self.get_next_open_row(column)
        if row is None:
            logger.error(f"No open rows available in column {column}.")
            return False

        self.board[row][column] = team
        self.last_team = team
        logger.debug(f"Team {team} placed in column {column}, row {row}.")
        return True

    def is_valid_move(self, column):
        """Check if a move is valid."""
        return (0 <= column < COLUMN_COUNT and 
                self.board[ROW_COUNT-1][column] == EMPTY)

    def get_next_open_row(self, column):
        """Get the next available row in the given column starting from the bottom."""
        for row in range(ROW_COUNT-1, -1, -1):
            if self.board[row][column] == EMPTY:
                return row
        return None

    def check_win(self, team):
        """Check if the given team has won."""
        # Check horizontal
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r][c+i] == team for i in range(4)):
                    return True

        # Check vertical
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT):
                if all(self.board[r+i][c] == team for i in range(4)):
                    return True

        # Check positive diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r+i][c+i] == team for i in range(4)):
                    return True

        # Check negative diagonal
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r-i][c+i] == team for i in range(4)):
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
        """Return string representation of board with colored circles."""
        mapping = {
            RED_TEAM: '\033[91m\u25CF\033[0m',  # Red circle
            YEL_TEAM: '\033[93m\u25CF\033[0m',  # Yellow circle
            EMPTY: '\u25CB'                      # White circle
        }
        # Reverse the board for correct visualization (bottom row first)
        return '\n'.join(' '.join(mapping[cell] for cell in row) for row in self.board[::-1])