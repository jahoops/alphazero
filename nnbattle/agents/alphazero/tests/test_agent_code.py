# /tests/test_agent_code.py

# Configure logging
import logging
import unittest
from unittest.mock import MagicMock, patch

from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent  # Updated absolute import
from nnbattle.agents.alphazero.network import Connect4Net      # Updated absolute import
from nnbattle.agents.alphazero.mcts import MCTSNode            # Ensure MCTSNode is correctly imported
from nnbattle.game.connect_four_game import ConnectFourGame 

# Configure logging if present
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG or other levels to INFO
logger = logging.getLogger(__name__)

class TestAlphaZeroAgent(unittest.TestCase):
    @patch('nnbattle.agents.alphazero.network.Connect4Net')
    @patch('nnbattle.agents.alphazero.mcts.MCTSNode')
    def setUp(self, mock_mcts_node, mock_connect4net):
        self.agent = AlphaZeroAgent(
            state_dim=2,  # Updated from previous value
            action_dim=7,
            use_gpu=False,
            model_path="/model/test_model.pth"
        )
        # Mock the neural network and MCTSNode
        self.agent.model = mock_connect4net.return_value
        self.agent.model.eval = MagicMock()
        mock_mcts_node.return_value = MagicMock()

    @patch('nnbattle.agents.alphazero.agent_code.torch.load')
    def test_load_model_success(self, mock_torch_load):
        # Simulate successful model loading
        mock_torch_load.return_value = {}
        self.agent.load_model()
        self.assertTrue(self.agent.model_loaded)
        self.agent.model.load_state_dict.assert_called()

    @patch('nnbattle.agents.alphazero.agent_code.torch.load')
    def test_load_model_failure(self, mock_torch_load):
        # Simulate model loading failure
        mock_torch_load.side_effect = Exception("Load failed")
        with self.assertRaises(Exception):
            self.agent.load_model()
        self.assertFalse(self.agent.model_loaded)

    def test_select_move_no_model_loaded(self):
        # Ensure model is loaded if not already
        with patch.object(self.agent, 'load_model') as mock_load_model:
            self.agent.model_loaded = False
            self.agent.mcts_simulate = MagicMock(return_value=(3, [3,4], [0.6, 0.4]))
            action = self.agent.select_move(ConnectFourGame())
            mock_load_model.assert_called_once()
            self.assertEqual(action, 3)

    def test_select_move_model_already_loaded(self):
        # Ensure model is not loaded again if already loaded
        with patch.object(self.agent, 'load_model') as mock_load_model:
            self.agent.model_loaded = True
            self.agent.mcts_simulate = MagicMock(return_value=(4, [2,5], [0.7, 0.3]))
            action = self.agent.select_move(ConnectFourGame())
            mock_load_model.assert_not_called()
            self.assertEqual(action, 4)

    def test_select_move_no_actions(self):
        # Simulate scenario where no actions are available
        self.agent.mcts_simulate = MagicMock(return_value=(None, [], []))
        action = self.agent.select_move(ConnectFourGame())
        self.assertIsNone(action)

    def test_select_move_different_channels(self):
        # Create a dummy state with 2 channels
        dummy_state = torch.randn(1, 2, 6, 7)  # [1, 2, 6, 7]
        with patch.object(self.agent, 'preprocess', return_value=dummy_state.squeeze(0)):  # Ensure [2,6,7]
            action = self.agent.select_move(ConnectFourGame())
            self.assertIsNotNone(action, "Action should be selected with correct input channels.")

    def test_self_play_deterministic(self):
        # Test self-play game data storage
        with patch.object(self.agent, 'select_move', return_value=3):
            with patch.object(self.agent.env, 'step', return_value=(np.zeros((6,7), dtype=np.int8), 0, True, {})):
                self.agent.self_play()
                self.assertEqual(len(self.agent.memory), 1)
                state, mcts_prob, value = self.agent.memory[0]
                self.assertEqual(mcts_prob[3], 1.0)
                self.assertEqual(value, -1)  # Assuming player 1 lost

if __name__ == '__main__':
    unittest.main()