# game/connect_four_game.py

import copy
import logging

import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PLAYER_PIECE = 1
# Removed duplicate PLAYER_PIECE definition
# PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)  # 6 rows x 7 columns
        self.current_player = PLAYER_PIECE  # Player 1 starts

    def copy(self):
        new_game = ConnectFourGame()
        new_game.board = self.board.copy()  # Ensure a copy of the NumPy array is made
        new_game.current_player = self.current_player
        return new_game

    def reset(self):
        """
        Resets the game to the initial state.
        """
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.current_player = PLAYER_PIECE
        return self.board.copy()

    def make_move(self, action):
        if self.is_valid_location(action):
            row = self.get_next_open_row(action)
            self.drop_piece(row, action, self.current_player)
            self.current_player = AI_PIECE if self.current_player == PLAYER_PIECE else PLAYER_PIECE
        else:
            logger.error(f"Invalid move attempted: Column {action} is full.")

    def step(self, action):
        """
        Applies the action to the board and returns the game status.
        """
        self.make_move(action)
        terminal = self.is_terminal()
        reward = self.get_reward()
        return self.board, reward, terminal

    def is_terminal(self):
        return self.check_win(PLAYER_PIECE) or self.check_win(AI_PIECE) or self.is_board_full()

    def get_reward(self):
        if self.check_win(PLAYER_PIECE):
            return 1  # Player wins
        elif self.check_win(AI_PIECE):
            return -1  # AI wins
        else:
            return 0  # Draw or ongoing

    def get_board_state(self):
        return self.board.copy()

    def is_terminal_node(self):
        return self.is_terminal()

    def get_valid_locations(self):
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations

    def is_valid_location(self, col):
        return self.board[ROW_COUNT-1][col] == EMPTY

    def get_next_open_row(self, col):
        for r in range(ROW_COUNT):
            if self.board[r][col] == EMPTY:
                return r
        return None  # Should not happen if checked with is_valid_location

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def check_win(self, piece):
        # Check horizontal locations
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT):
                if all(self.board[r][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Check vertical locations
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMN_COUNT-3):
            for r in range(3, ROW_COUNT):
                if all(self.board[r-i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        return False

    def get_win_positions(self, piece):
        win_positions = []
        # Similar checks as check_win but store winning positions
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT):
                if all(self.board[r][c+i] == piece for i in range(WINDOW_LENGTH)):
                    win_positions.append([(r, c+i) for i in range(WINDOW_LENGTH)])

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c] == piece for i in range(WINDOW_LENGTH)):
                    win_positions.append([(r+i, c) for i in range(WINDOW_LENGTH)])

        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    win_positions.append([(r+i, c+i) for i in range(WINDOW_LENGTH)])

        for c in range(COLUMN_COUNT-3):
            for r in range(3, ROW_COUNT):
                if all(self.board[r-i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    win_positions.append([(r-i, c+i) for i in range(WINDOW_LENGTH)])

        return win_positions

    def is_board_full(self):
        return all(self.board[ROW_COUNT-1][c] != EMPTY for c in range(COLUMN_COUNT))

    def get_winner(self):
        if self.check_win(PLAYER_PIECE):
            return PLAYER_PIECE
        elif self.check_win(AI_PIECE):
            return AI_PIECE
        else:
            return 0  # No winner

    def score_position(self, piece):
        score = 0
        # Score horizontal
        for r in range(ROW_COUNT):
            row_array = list(self.board[r])
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)
        # Score vertical
        for c in range(COLUMN_COUNT):
            col_array = list(self.board[:,c])
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)
        # Score positive sloped diagonals
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)
        # Score negative sloped diagonals
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)
        return score

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4

        return score

    def get_result(self):
        return self.get_winner()

    def get_state(self):
        return self.board.copy()

    def board_to_string(self):
        mapping = {PLAYER_PIECE: 'X', AI_PIECE: 'O', EMPTY: '.'}
        rows = []
        for row in self.board:
            rows.append(' '.join(mapping[cell] for cell in row))
        return '\n'.join(rows)