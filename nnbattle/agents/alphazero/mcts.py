# /mcts.py

import logging
import math
import numpy as np
from typing import Optional, List
from copy import deepcopy

import torch

from nnbattle.game.connect_four_game import ConnectFourGame 
from nnbattle.agents.alphazero.utils.model_utils import preprocess_board

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], env: ConnectFourGame):
        self.parent = parent
        self.action = action
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
                new_env = self.env.copy()
                new_env.make_move(action)
                new_env.current_player = 2 if new_env.current_player == 1 else 1
                child_node = MCTSNode(parent=self, action=action, env=new_env)
                child_node.prior = action_probs[action].item()
                self.children[action] = child_node

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

__all__ = ['MCTSNode']

# No changes needed if not mocking base classes
# Ensure that in tests, only methods are mocked, not entire classes that use super()