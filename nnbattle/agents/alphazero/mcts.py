# /mcts.py

import logging
import math
import numpy as np
from typing import Optional, List
import copy

import torch

from nnbattle.game.connect_four_game import ConnectFourGame 
from nnbattle.constants import RED_TEAM, YEL_TEAM  # Ensure constants are imported
from nnbattle.agents.alphazero.utils.model_utils import preprocess_board

logger = logging.getLogger(__name__)

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], env: ConnectFourGame):
        self.parent = parent
        self.action = action
        # Determine the team that will make the next move
        if env.last_piece is None:
            self.team = RED_TEAM  # Starting team
        else:
            self.team = 3 - env.last_piece  # Switch team
        self.env = env
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
        for action in legal_actions:
            if action not in self.children:
                new_env = deepcopy_env(self.env)
                new_env.make_move(action, self.team)  # self.team is now valid
                child_node = MCTSNode(parent=self, action=action, env=new_env)
                child_node.prior = action_probs[action].item()
                self.children[action] = child_node

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

def mcts_simulate(agent, game: ConnectFourGame, valid_moves):
    logger.info("Starting MCTS simulation...")
    root = MCTSNode(parent=None, action=None, env=deepcopy_env(game))
    root.visits = 1

    for _ in range(agent.num_simulations):
        node = root
        env = deepcopy_env(game)

        # **Selection**
        while not node.is_leaf():
            node = node.best_child(agent.c_puct)
            if node is None:
                break
            env.make_move(node.action, node.team)

        # **Expansion**
        if env.get_game_state() == "ONGOING":
            state = agent.preprocess(env.get_board())
            action_probs, value = agent.model(state.unsqueeze(0))
            action_probs = action_probs.squeeze().detach().cpu()
            value = value.item()
            # Mask invalid moves
            valid_actions = env.get_valid_moves()
            action_mask = torch.zeros(agent.action_dim)
            action_mask[valid_actions] = 1
            action_probs *= action_mask
            if action_probs.sum() > 0:
                action_probs /= action_probs.sum()
            else:
                # If all probabilities are zero, assign equal probability to valid actions
                action_probs[valid_actions] = 1.0 / len(valid_actions)
            node.expand(action_probs, valid_actions)
            reward = value  # **Use the network's value prediction as reward**

        else:
            # **Terminal State**
            final_state = env.get_game_state()
            if final_state == "WIN":
                winner = env.last_piece
                reward = 1.0 if winner == agent.team else -1.0
            else:
                reward = 0.0  # Draw

        # **Backpropagation**
        node.backpropagate(reward)

    # **Selection of the Best Action**
    action_visits = [(child.action, child.visits) for child in root.children.values()]
    if not action_visits:
        raise InvalidMoveError("No valid moves found during MCTS.")

    selected_action = max(action_visits, key=lambda x: x[1])[0]
    action_probs = torch.zeros(agent.action_dim)
    for action, visits in action_visits:
        action_probs[action] = visits
    action_probs /= action_probs.sum()

    return selected_action, action_probs

__all__ = ['MCTSNode']

# No changes needed if not mocking base classes
# Ensure that in tests, only methods are mocked, not entire classes that use super()