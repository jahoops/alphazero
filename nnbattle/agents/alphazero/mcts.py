# /mcts.py

import logging
import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM
from .utils.model_utils import preprocess_board

logger = logging.getLogger(__name__)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], board: np.ndarray, team: int):
        """Initialize node with board state instead of game instance."""
        self.parent = parent
        self.action = action
        self.board = board.copy()  # Store board state instead of game instance
        self.team = team
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c_puct: float) -> Optional['MCTSNode']:
        best_score = -float('inf')
        best_node = None
        for child in self.children.values():
            u = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            q = child.value / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self, action_probs: torch.Tensor, legal_actions: List[int]):
        """
        Expand node with valid moves from the game state.
        """
        # Get valid moves directly from the environment
        valid_moves = self.env.get_valid_moves()
        for action in valid_moves:  # Only iterate over valid moves
            if action not in self.children:
                new_env = deepcopy_env(self.env)
                try:
                    # Make move with current team
                    new_env.make_move(action, self.team)
                    # Next team is opposite of last_team
                    next_team = YEL_TEAM if new_env.last_team == RED_TEAM else RED_TEAM
                    child_node = MCTSNode(parent=self, action=action, env=new_env, team=next_team)
                    child_node.prior = action_probs[action].item()
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
    """Monte Carlo Tree Search simulation."""
    if game.last_team == team:
        raise InvalidTurnError(f"Invalid turn: team {team} cannot move after itself")

    # Get initial state and valid moves
    board = game.get_board()
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        raise InvalidMoveError("No valid moves available")

    # Save the original mode
    original_mode = agent.model.training
    agent.model.eval()

    try:
        # Initialize root with board state
        root = MCTSNode(None, None, board, team)
        root.visits = 1

        # Run simulations
        for _ in range(agent.num_simulations):
            # Create new game instance for this simulation
            sim_game = ConnectFourGame()
            sim_game.board = root.board.copy()
            sim_game.last_team = game.last_team
            
            node = root
            current_team = team

            # Selection
            while not node.is_leaf():
                node = node.best_child(agent.c_puct)
                if node is None:
                    break
                # Apply action to simulation game
                sim_game.make_move(node.action, node.team)
                current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM

            # Expansion
            if sim_game.get_game_state() == "ONGOING" and node.is_leaf():
                # Get valid moves for current state
                sim_valid_moves = sim_game.get_valid_moves()
                if sim_valid_moves:
                    # Get network predictions
                    with torch.no_grad():
                        state_tensor = agent.preprocess(sim_game.get_board(), current_team, to_device=agent.device)
                        action_logits, value = agent.model(state_tensor.unsqueeze(0))
                        action_probs = F.softmax(action_logits.squeeze(), dim=0)

                    # Mask invalid moves and normalize
                    action_mask = torch.zeros_like(action_probs)
                    action_mask[sim_valid_moves] = 1
                    masked_probs = action_probs * action_mask
                    if masked_probs.sum() > 0:
                        masked_probs /= masked_probs.sum()

                    # Create child nodes
                    for move in sim_valid_moves:
                        new_game = ConnectFourGame()
                        new_game.board = sim_game.get_board().copy()
                        new_game.last_team = sim_game.last_team
                        if new_game.make_move(move, current_team):
                            child = MCTSNode(
                                parent=node,
                                action=move,
                                board=new_game.get_board(),
                                team=(YEL_TEAM if current_team == RED_TEAM else RED_TEAM)
                            )
                            child.prior = masked_probs[move].item()
                            node.children[move] = child

            # Backpropagation
            game_result = sim_game.get_game_state()
            if game_result == "ONGOING":
                value = value.item()
            else:
                value = 1.0 if game_result == team else (-1.0 if game_result in (RED_TEAM, YEL_TEAM) else 0.0)
            node.backpropagate(-value)  # Negative because we're alternating perspectives

        # Select move based on visit counts and temperature
        valid_children = [(child.action, child.visits) for child in root.children.values()]
        if not valid_children:
            return np.random.choice(valid_moves), torch.zeros(agent.action_dim)

        actions, visits = zip(*valid_children)
        visits = np.array(visits)

        if temperature < 0.01:
            action = actions[np.argmax(visits)]
        else:
            probs = visits ** (1.0 / temperature)
            probs = probs / probs.sum()
            action = np.random.choice(actions, p=probs)

        # Return selected action and normalized visit counts as policy
        policy = torch.zeros(agent.action_dim)
        policy[list(actions)] = torch.tensor(visits / visits.sum())
        return action, policy

    finally:
        agent.model.train(original_mode)

__all__ = ['MCTSNode', 'mcts_simulate']

# No changes needed if not mocking base classes
# Ensure that in tests, only methods are mocked, not entire classes that use super()