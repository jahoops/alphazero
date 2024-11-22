# /mcts.py

import logging
import math
from copy import deepcopy

from nnbattle.game.connect_four_game import ConnectFourGame 

from .utils import deepcopy_env  # Assuming you have a deepcopy utility

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, parent=None, action=None, env=None):
        self.parent = parent
        self.action = action
        self.env = deepcopy_env(env) if env else None
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c_puct=1.4):
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = (child.value / (child.visits + 1)) + c_puct * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, action_probs, legal_moves):
        for action in legal_moves:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, action=action, env=self.env)

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

__all__ = ['MCTSNode']