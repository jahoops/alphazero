# FILE: minimax_agent/agent_code.py
import math
import random

ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

class ConnectFourGame:
    def __init__(self):
        self.board = self.create_board()

    def create_board(self):
        return [[EMPTY for _ in range(COLUMN_COUNT)] for _ in range(ROW_COUNT)]

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[0][col] == EMPTY

    def get_next_open_row(self, col):
        for r in range(ROW_COUNT-1, -1, -1):
            if self.board[r][col] == EMPTY:
                return r
        return None

    def winning_move(self, piece):
        # Check horizontal locations
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Check vertical locations
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Check positively sloped diagonals
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r+i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        # Check negatively sloped diagonals
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r-i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        return False

    def get_valid_locations(self):
        return [c for c in range(COLUMN_COUNT) if self.is_valid_location(c)]

    def is_terminal_node(self):
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or len(self.get_valid_locations()) == 0

    def score_position(self, piece):
        score = 0

        # Score center column
        center_array = [int(row[COLUMN_COUNT//2]) for row in self.board]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in self.board[r]]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(self.board[r][c]) for r in range(ROW_COUNT)]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonals
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        # Score negative sloped diagonals
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
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

    def copy(self):
        new_game = ConnectFourGame()
        new_game.board = [row.copy() for row in self.board]
        return new_game

class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def select_move(self, game):
        valid_locations = game.get_valid_locations()
        if not valid_locations:
            return None
        score, column = self.minimax(game, self.depth, -math.inf, math.inf, True)
        return column

    def minimax(self, game, depth, alpha, beta, maximizingPlayer):
        valid_locations = game.get_valid_locations()
        is_terminal = game.is_terminal_node()
        if depth == 0 or is_terminal:
            if is_terminal:
                if game.winning_move(AI_PIECE):
                    return (math.inf, None)
                elif game.winning_move(PLAYER_PIECE):
                    return (-math.inf, None)
                else:
                    return (0, None)
            else:
                return (game.score_position(AI_PIECE), None)

        if maximizingPlayer:
            value = -math.inf
            best_column = random.choice(valid_locations)
            for col in valid_locations:
                row = game.get_next_open_row(col)
                temp_game = game.copy()
                temp_game.drop_piece(row, col, AI_PIECE)
                new_score, _ = self.minimax(temp_game, depth-1, alpha, beta, False)
                if new_score > value:
                    value = new_score
                    best_column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_column
        else:
            value = math.inf
            best_column = random.choice(valid_locations)
            for col in valid_locations:
                row = game.get_next_open_row(col)
                temp_game = game.copy()
                temp_game.drop_piece(row, col, PLAYER_PIECE)
                new_score, _ = self.minimax(temp_game, depth-1, alpha, beta, True)
                if new_score < value:
                    value = new_score
                    best_column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_column