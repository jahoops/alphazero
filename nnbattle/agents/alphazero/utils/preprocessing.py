
import copy
import numpy as np
import torch

def preprocess_board(board, player):
    """
    Preprocesses the board by applying the player's perspective.

    :param board: Current game board as a NumPy array.
    :param player: Current player (1 or -1).
    :return: Preprocessed board as a Torch tensor.
    """
    return copy.deepcopy(board) * player