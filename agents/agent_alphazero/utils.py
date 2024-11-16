# alphazero_agent/utils.py

import copy

def preprocess_board(board, player):
    """
    Preprocesses the board by applying the player's perspective.

    :param board: Current game board as a NumPy array.
    :param player: Current player (1 or -1).
    :return: Preprocessed board as a Torch tensor.
    """
    return copy.deepcopy(board) * player

def deepcopy_env(env):
    """
    Creates a deep copy of the game environment.

    :param env: Current game environment.
    :return: A deep copy of the game environment.
    """
    return copy.deepcopy(env)