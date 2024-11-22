# /mcts.py

import logging
import math
from copy import deepcopy

from nnbattle.game.connect_four_game import ConnectFourGame 

from .utils import deepcopy_env  # Assuming you have a deepcopy utility

logger = logging.getLogger(__name__)

# Removed MCTSNode class as it's now in agent_code.py