# agents/agent_minimax/agent_code.py

import math
import random
import logging
import copy
from nnbattle.agents.base_agent import Agent

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from nnbattle.game.connect_four_game import ConnectFourGame 
from nnbattle.constants import RED_TEAM, YEL_TEAM

class MinimaxAgent(Agent):
    def __init__(self, depth=4, team=YEL_TEAM):
        """
        Initializes the MinimaxAgent with a specified search depth and team number.
        
        :param depth: The depth to which the Minimax algorithm will search.
        :param team: The team number (1 or 2) that the agent is playing for.
        """
        self.depth = depth
        self.team = team  # Assign team to the agent
        logger.info(f"MinimaxAgent initialized with team {self.team} and depth {self.depth}")

    def select_move(self, game: ConnectFourGame):
        """
        Selects the best move by running the Minimax algorithm with Alpha-Beta pruning.
        
        :param game: The current state of the game.
        :return: The column number (0-6) where the agent decides to drop its piece.
        """
        score, column = self.minimax(game, self.depth, -math.inf, math.inf, True)
        logger.info(f"Selected move: {column} with score: {score}")
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
        valid_moves = game.get_valid_moves()
        result = game.get_game_state()
        if result != "ONGOING":
            if result == self.team:
                return (math.inf, None)
            elif result == 3 - self.team:
                return (-math.inf, None)
            else:  # Game is over, no more valid moves
                return (0, None)
        elif depth == 0:  # Depth is zero
            return (game.score_position(3 - self.team), None)
        else:
            if maximizingPlayer:
                value = -math.inf
                best_column = random.choice(valid_moves)
                for col in valid_moves:
                    temp_game = copy.deepcopy(game)
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
                best_column = random.choice(valid_moves)
                for col in valid_moves:
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