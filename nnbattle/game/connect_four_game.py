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
        """
        Applies an action to the game state.
        Action represents the column (0-6) where the current player drops a piece.
        
        :param action: Integer representing the column index.
        :return: Boolean indicating if the move was successful.
        """
        if self.is_valid_location(action):
            for row in range(ROW_COUNT-1, -1, -1):
                if self.board[row][action] == EMPTY:
                    self.board[row][action] = self.current_player
                    self.current_player = AI_PIECE if self.current_player == PLAYER_PIECE else PLAYER_PIECE
                    return True
        return False

    def step(self, action):
        """
        Applies an action and returns the new state, reward, done status, and additional info.
        
        :param action: Integer representing the column index.
        :return: Tuple (new_board, reward, done, info)
        """
        move_successful = self.make_move(action)
        if not move_successful:
            # Invalid move attempt
            reward = -10.0  # Penalty for invalid move
            done = True
            return self.board.copy(), reward, done, {"invalid_move": True}
        
        done = self.is_terminal()
        reward = self.get_reward()
        return self.board.copy(), reward, done, {}

    def is_terminal(self):
        """
        Checks if the game has ended either by win or draw.
        
        :return: Boolean indicating if the game is over.
        """
        return self.check_win(PLAYER_PIECE) or self.check_win(AI_PIECE) or self.is_board_full()

    def get_reward(self):
        """
        Returns the reward based on the current game state.
        
        :return: Float representing the reward.
        """
        if self.check_win(AI_PIECE):
            return 1.0  # AI wins
        elif self.check_win(PLAYER_PIECE):
            return -1.0  # Player wins
        else:
            return 0.0  # Draw or ongoing game

    def get_board_state(self):
        """
        Returns the current state of the board.
        
        :return: Numpy array representing the board state.
        """
        return self.board.copy()

    def is_terminal_node(self):
        """
        Checks if the game has reached a terminal state.
        
        :return: Boolean indicating if the game is over.
        """
        return self.is_terminal()

    def get_valid_locations(self):
        """
        Returns a list of valid actions (columns that are not full).
        
        :return: List of integers representing valid column indices.
        """
        return [col for col in range(COLUMN_COUNT) if self.is_valid_location(col)]

    def is_valid_location(self, col):
        """
        Checks if the top row of the specified column is empty.
        
        :param col: Integer representing the column index.
        :return: Boolean indicating if a move can be made in the column.
        """
        assert isinstance(self.board, np.ndarray), f"self.board is not a NumPy array but {type(self.board)}"
        return self.board[0][col] == EMPTY

    def get_next_open_row(self, col):
        """
        Returns the next open row in the specified column.
        
        :param col: Integer representing the column index.
        :return: Integer representing the row index.
        """
        for row in range(ROW_COUNT-1, -1, -1):
            if self.board[row][col] == EMPTY:
                return row

    def drop_piece(self, row, col, piece):
        """
        Drops a piece into the specified location on the board.
        
        :param row: Integer representing the row index.
        :param col: Integer representing the column index.
        :param piece: Integer representing the player's piece.
        """
        self.board[row][col] = piece

    def check_win(self, player):
        """
        Checks if the specified player has won the game.
        
        :param player: Integer representing the player's piece.
        :return: Boolean indicating if the player has won.
        """
        win_positions = self.get_win_positions(player)
        if win_positions:
            logger.info(f"Player {player} has winning positions: {win_positions}")
        return len(win_positions) > 0

    def get_win_positions(self, player):
        """
        Identifies all winning positions for a given player.
        
        :param player: Integer representing the player's piece.
        :return: List of winning positions.
        """
        wins = []
        # Horizontal
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                window = self.board[row, col:col+WINDOW_LENGTH]
                if np.all(window == player):
                    wins.append(((row, col), (0, 1)))
        # Vertical
        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT - 3):
                window = self.board[row:row+WINDOW_LENGTH, col]
                if np.all(window == player):
                    wins.append(((row, col), (1, 0)))
        # Positive Diagonal
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT - 3):
                window = [self.board[row+i][col+i] for i in range(WINDOW_LENGTH)]
                if all(cell == player for cell in window):
                    wins.append(((row, col), (1, 1)))
        # Negative Diagonal
        for row in range(3, ROW_COUNT):  # Changed range from (ROW_COUNT - 3) to (3, ROW_COUNT)
            for col in range(COLUMN_COUNT - 3):
                window = [self.board[row - i][col + i] for i in range(WINDOW_LENGTH)]  # Corrected indexing
                if all(cell == player for cell in window):
                    wins.append(((row, col), (-1, 1)))
        return wins

    def is_board_full(self):
        """
        Checks if the board is full.
        
        :return: Boolean indicating if the board is full.
        """
        return not (self.board == EMPTY).any()

    def get_winner(self):
        """
        Determines the winner of the game.
        
        :return: Integer representing the winner (PLAYER_PIECE, AI_PIECE, or EMPTY for no winner).
        """
        if self.check_win(AI_PIECE):
            return AI_PIECE
        elif self.check_win(PLAYER_PIECE):
            return PLAYER_PIECE
        else:
            return EMPTY

    def score_position(self, piece):
        """
        Scores the board position based on the number of potential winning opportunities.
        
        :param piece: Integer representing the player's piece.
        :return: Integer score representing the desirability of the position.
        """
        score = 0

        # Center column preference
        center_array = [int(i) for i in list(self.board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for row in range(ROW_COUNT):
            row_array = [int(i) for i in list(self.board[row,:])]
            for col in range(COLUMN_COUNT-3):
                window = row_array[col:col+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for col in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:,col])]
            for row in range(ROW_COUNT-3):
                window = col_array[row:row+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonals
        for row in range(ROW_COUNT-3):
            for col in range(COLUMN_COUNT-3):
                window = [self.board[row+i][col+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        # Score negative sloped diagonals
        for row in range(3, ROW_COUNT):  # Changed from range(ROW_COUNT-3) to range(3, ROW_COUNT)
            for col in range(COLUMN_COUNT-3):
                window = [self.board[row-i][col+i] for i in range(WINDOW_LENGTH)]  # Corrected indexing
                score += self.evaluate_window(window, piece)

        return score

    def evaluate_window(self, window, piece):
        """
        Evaluates a window of four cells and assigns a score.
        
        :param window: List of four integers representing a window on the board.
        :param piece: Integer representing the player's piece.
        :return: Integer score for the window.
        """
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
        """Get the game result.
        Returns:
            1: player 1 wins
            -1: player 2 wins
            0: draw
            None: game not finished
        """
        if self.is_terminal():
            # Check win for both players
            if self.check_win(1):
                return 1
            elif self.check_win(2):
                return -1
            # If no winner and terminal, it's a draw
            return 0
        return None

    def get_state(self):
        """Get the current state of the game board.
        
        Returns:
            numpy.ndarray: Current board state
        """
        return self.board.copy()  # Return a copy to prevent external modifications