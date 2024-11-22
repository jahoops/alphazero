# /tests/test_agent_code.py

import logging
import unittest
from unittest.mock import MagicMock, patch

from nnbattle.agents.alphazero.utils import initialize_agent, load_agent_model, save_agent_model  # Updated imports
from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent  # Ensure correct import
from nnbattle.game.connect_four_game import ConnectFourGame  # ...existing imports...

# Configure logging if present
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAlphaZeroAgent(unittest.TestCase):
    # Use autospec=True to ensure mocks retain type information
    @patch('nnbattle.agents.alphazero.network.Connect4Net', autospec=True)
    @patch('nnbattle.agents.alphazero.mcts.MCTSNode', autospec=True)
    def setUp(self, mock_mcts_node, mock_connect4net):
        # Initialize agent using initialize_agent without specifying model_path
        self.agent = initialize_agent(
            state_dim=2,
            action_dim=7,
            use_gpu=False,
            # model_path="/model/test_model.pth",  # Removed to use MODEL_PATH from utils.py
            num_simulations=800,
            c_puct=1.4,
            load_model=False  # Set to False to avoid loading during tests
        )
        # Mock the neural network and MCTSNode
        self.agent.model = mock_connect4net.return_value
        self.agent.model.eval = MagicMock()
        mock_mcts_node.return_value = MagicMock()

    @patch('nnbattle.agents.alphazero.utils.load_agent_model')  # Patch load_agent_model
    @patch('nnbattle.agents.alphazero.agent_code.torch.load')
    def test_load_model_success(self, mock_torch_load, mock_load_agent_model):
        # Simulate successful model loading
        mock_load_agent_model.return_value = None  # Assuming load_agent_model doesn't return anything
        load_agent_model(self.agent)
        mock_load_agent_model.assert_called_with(self.agent)
        # ...additional assertions if needed...

    @patch('nnbattle.agents.alphazero.utils.load_agent_model', side_effect=Exception("Load failed"))
    @patch('nnbattle.agents.alphazero.agent_code.torch.load')
    def test_load_model_failure(self, mock_torch_load, mock_load_agent_model):
        # Simulate model loading failure
        with self.assertRaises(Exception):
            load_agent_model(self.agent)
        # ...additional assertions if needed...

    @patch('nnbattle.agents.alphazero.utils.save_agent_model')  # Patch save_agent_model
    def test_save_model_success(self, mock_save_agent_model):
        save_agent_model(self.agent)  # Removed the path parameter
        mock_save_agent_model.assert_called_with(self.agent)
        mock_logger.info.assert_called_with("Model saved to nnbattle/agents/alphazero/model/alphazero_model_final.pth.")
    
    @patch('nnbattle.agents.alphazero.utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_model_failure(self, mock_save_agent_model):
        # Simulate model saving failure
        with self.assertRaises(Exception):
            save_agent_model(self.agent)  # Removed the path parameter
        mock_logger.error.assert_called_with("Error saving model: Save failed")
    
    def test_select_move_no_model_loaded(self):
        # Ensure model is loaded if not already
        with patch('nnbattle.agents.alphazero.utils.load_agent_model') as mock_load_agent_model:
            self.agent.model_loaded = False
            self.agent.mcts_simulate = MagicMock(return_value=(3, [3,4], [0.6, 0.4]))
            action, action_probs = self.agent.select_move(ConnectFourGame())
            mock_load_agent_model.assert_called_once_with(self.agent)
            self.assertEqual(action, 3)

    def test_select_move_model_already_loaded(self):
        # Ensure model is not loaded again if already loaded
        with patch('nnbattle.agents.alphazero.utils.load_agent_model') as mock_load_agent_model:
            self.agent.model_loaded = True
            self.agent.mcts_simulate = MagicMock(return_value=(4, [2,5], [0.7, 0.3]))
            action, action_probs = self.agent.select_move(ConnectFourGame())
            mock_load_agent_model.assert_not_called()
            self.assertEqual(action, 4)

    def test_select_move_no_actions(self):
        # Simulate scenario where no actions are available
        self.agent.mcts_simulate = MagicMock(return_value=(None, [], []))
        action, action_probs = self.agent.select_move(ConnectFourGame())
        self.assertIsNone(action)

    def test_select_move_different_channels(self):
        # Create a dummy state with 2 channels
        dummy_state = torch.randn(1, 2, 6, 7)  # [1, 2, 6, 7]
        with patch.object(self.agent, 'preprocess', return_value=dummy_state.squeeze(0)):  # Ensure [2,6,7]
            action, action_probs = self.agent.select_move(ConnectFourGame())
            self.assertIsNotNone(action, "Action should be selected with correct input channels.")

    def test_self_play_deterministic(self):
        self.agent.mcts_simulate = MagicMock(return_value=(None, [], []))
        with patch.object(self.agent, 'select_move', return_value=(3, torch.tensor([0,0,0,1,0,0,0], dtype=torch.float32))):
            with patch.object(self.agent.env, 'step', return_value=(np.zeros((6,7), dtype=np.int8), 0, True, {})):
                self.agent.self_play()
                self.assertEqual(len(self.agent.memory), 1)
                state, mcts_prob, value = self.agent.memory[0]
                self.assertEqual(mcts_prob[3], 1.0)
                self.assertEqual(value, -1)  # Assuming player 1 lost

    @patch('nnbattle.agents.alphazero.utils.save_agent_model')  # Patch save_agent_model
    def test_save_model_success(self, mock_save_agent_model):
        # Simulate successful model saving
        save_agent_model(self.agent, "model/path.pth")
        mock_save_agent_model.assert_called_with(self.agent, "model/path.pth")

    @patch('nnbattle.agents.alphazero.utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_model_failure(self, mock_save_agent_model):
        # Simulate model saving failure
        with self.assertRaises(Exception):
            save_agent_model(self.agent, "model/path.pth")
        # ...additional assertions if needed...

if __name__ == '__main__':
    unittest.main()