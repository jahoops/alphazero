# agents/agent_minimax/agent_code.py

import math
import random

from nnbattle.agents.base_agent import Agent
from nnbattle.game.connect_four_game import ConnectFourGame 

class MinimaxAgent(Agent):
    def __init__(self, depth=4, team=1):
        """
        Initializes the MinimaxAgent with a specified search depth and team number.
        
        :param depth: The depth to which the Minimax algorithm will search.
        :param team: The team number (1 or 2) that the agent is playing for.
        """
        self.depth = depth
        self.team = 1

    def select_move(self, game: ConnectFourGame):
        """
        Selects the best move by running the Minimax algorithm with Alpha-Beta pruning.
        
        :param game: The current state of the game.
        :return: The column number (0-6) where the agent decides to drop its piece.
        """
        valid_locations = game.get_valid_locations()
        if not valid_locations:
            return None
        score, column = self.minimax(game, self.depth, -math.inf, math.inf, True)
        return column

    def minimax(self, game: ConnectFourGame, depth, alpha, beta, maximizingPlayer):
        """
        The Minimax algorithm with Alpha-Beta pruning.
        
        :param game: The current game state.
        :param depth: The current depth in the game tree.
        :param alpha: The alpha value for pruning.
        :param beta: The beta value for pruning.
        :param maximizingPlayer: Boolean indicating if the current layer is maximizing or minimizing.
        :return: Tuple of (score, column)
        """
        valid_locations = game.get_valid_locations()
        is_terminal = game.is_terminal()
        if depth == 0 or is_terminal:
            if is_terminal:
                if game.check_win(AI_PIECE):
                    return (math.inf, None)
                elif game.check_win(PLAYER_PIECE):
                    return (-math.inf, None)
                else:  # Game is over, no more valid moves
                    return (0, None)
            else:  # Depth is zero
                return (game.score_position(AI_PIECE), None)

        if maximizingPlayer:
            value = -math.inf
            best_column = random.choice(valid_locations)
            for col in valid_locations:
                temp_game = game.new_game()
                move_successful = temp_game.make_move(col)
                if not move_successful:
                    continue  # Skip invalid moves
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
                temp_game = game.new_game()
                move_successful = temp_game.make_move(col)
                if not move_successful:
                    continue  # Skip invalid moves
                new_score, _ = self.minimax(temp_game, depth-1, alpha, beta, True)
                if new_score < value:
                    value = new_score
                    best_column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_column