# /mcts.py

import logging
import math
import numpy as np
from typing import Optional
from copy import deepcopy

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

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c_puct: float) -> Optional['MCTSNode']:
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = (child.value / (child.visits + 1)) + c_puct * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, policy: np.ndarray, legal_moves: list):
        for action in legal_moves:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, action=action, env=self.env)

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

__all__ = ['MCTSNode']

# No changes needed if not mocking base classes
# Ensure that in tests, only methods are mocked, not entire classes that use super()