# FILE: agents/base_agent.py

from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def select_move(self, board):
        """
        Given the current board state, return the column number (0-6) where the agent wants to drop its piece.
        """
        pass