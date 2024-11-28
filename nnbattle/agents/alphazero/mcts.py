# /mcts.py

import logging
import math
import copy
from typing import Optional, List
import signal
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
import numpy as np
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM
from .utils.model_utils import preprocess_board

logger = logging.getLogger(__name__)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], board: np.ndarray, player: int):
        """Initialize node with board state and player whose turn it is."""
        self.parent = parent
        self.action = action
        self.board = board.copy()
        self.player = player  # The player to move at this node
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c_puct: float) -> Optional['MCTSNode']:
        best_score = -float('inf')
        best_node = None
        epsilon = 1e-8  # Small value to prevent division by zero
        for child in self.children.values():
            # Adjust denominator to add epsilon
            u = c_puct * child.prior * math.sqrt(self.visits + epsilon) / (1 + child.visits + epsilon)
            q = child.value / (1 + child.visits + epsilon)
            score = q + u
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self, action_probs: torch.Tensor, legal_actions: List[int]):
        """
        Expand node with valid moves from the game state.
        """
        for action in legal_actions:
            if action not in self.children:
                try:
                    next_board = self.board.copy()
                    # Simulate the action
                    next_game = ConnectFourGame()
                    next_game.board = next_board
                    # Do not set next_game.last_team manually
                    next_game.make_move(action, self.player)
                    # Switch player for the child node
                    next_player = YEL_TEAM if self.player == RED_TEAM else RED_TEAM
                    child_node = MCTSNode(
                        parent=self,
                        action=action,
                        board=next_game.get_board(),
                        player=next_player
                    )
                    child_node.prior = action_probs[action]
                    self.children[action] = child_node
                except (InvalidMoveError, InvalidTurnError) as e:
                    logger.error(f"Invalid move during expansion: {e}")
                    continue

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

def mcts_simulate(agent, game: ConnectFourGame, team: int, temperature=1.0):
    """Monte Carlo Tree Search simulation with early-stage optimization."""
    valid_moves = game.get_valid_moves()
    
    # Early in training, use mostly random play
    if agent.num_simulations < 25:
        action = np.random.choice(valid_moves)
        policy = torch.zeros(agent.action_dim, dtype=torch.float32, device=agent.device)
        policy[valid_moves] = 1.0 / len(valid_moves)
        return action, policy

    # Regular MCTS for when the model is more trained
    try:
        if game.last_team == team:
            raise InvalidTurnError(f"Invalid turn: team {team} cannot move after itself")

        # Create the root node with the current player
        root = MCTSNode(None, None, game.get_board(), player=team)
        root.visits = 1

        for sim in range(agent.num_simulations):
            current_node = root
            sim_game = deepcopy_env(game)

            # Selection
            while not current_node.is_leaf():
                current_node = current_node.best_child(agent.c_puct)
                if current_node is None:
                    break
                sim_game.make_move(current_node.action, current_node.player)

            # Expansion
            if sim_game.get_game_state() == "ONGOING":
                with torch.no_grad():
                    state_tensor = agent.preprocess(
                        sim_game.get_board(), current_node.player, to_device=agent.device
                    )
                action_logits, value = agent.model(state_tensor.unsqueeze(0))
                action_probs = F.softmax(action_logits.squeeze(), dim=0)
                valid_moves = sim_game.get_valid_moves()
                current_node.expand(action_probs, valid_moves)

            # Simulation / Evaluation
            game_result = sim_game.get_game_state()
            if game_result == "ONGOING":
                leaf_value = value.item()
            else:
                if game_result == team:
                    leaf_value = 1.0
                elif game_result == "Draw":
                    leaf_value = 0.0
                else:
                    leaf_value = -1.0

            # Backpropagation
            current_node.backpropagate(-leaf_value)

        # Select move based on visit counts and temperature
        valid_children = [(child.action, child.visits) for child in root.children.values()]
        if not valid_children:
            valid_moves = game.get_valid_moves()
            return np.random.choice(valid_moves), torch.zeros(agent.action_dim)

        actions, visits = zip(*valid_children)
        visits = np.array(visits, dtype=np.float32)

        # Ensure total visits are not zero to prevent division by zero
        total_visits = visits.sum()
        if total_visits == 0:
            logger.error("Total visits are zero, cannot compute probabilities")
            visits += 1e-8  # Add a small value to prevent zero division
            total_visits = visits.sum()

        # Compute probabilities safely
        probs = visits ** (1.0 / max(temperature, 1e-8))  # Prevent division by zero
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs_sum):
            logger.error("Sum of probabilities is zero or NaN, assigning uniform probabilities")
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        # Select action based on computed probabilities
        action = np.random.choice(actions, p=probs)

        # Create policy tensor on the correct device
        policy = torch.zeros(agent.action_dim, dtype=torch.float32, device=agent.device)
        policy[list(actions)] = torch.tensor(visits / total_visits, dtype=torch.float32, device=agent.device)

        return action, policy

    except Exception as e:
        logger.error(f"MCTS simulation failed: {str(e)}")
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        return np.random.choice(valid_moves), torch.zeros(agent.action_dim)
