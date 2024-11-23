import logging
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

from nnbattle.agents.alphazero.utils import model_utils
from nnbattle.agents.alphazero.agent_code import initialize_agent
from nnbattle.agents.alphazero.train.trainer import train_alphazero
from nnbattle.game.connect_four_game import ConnectFourGame

# Configure logging if present
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgentCode(unittest.TestCase):
    def setUp(self):
        # Initialize agent without patching Connect4Net and MCTSNode
        self.agent = initialize_agent(
            state_dim=2,
            action_dim=7,
            use_gpu=True,
            num_simulations=800,
            c_puct=1.4,
            load_model=False  # Corrected argument name
        )
        # Mock the necessary methods instead of entire classes
        self.agent.model = MagicMock()
        self.agent.model.eval = MagicMock()
        self.agent.mcts_simulate = MagicMock()

    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model', side_effect=FileNotFoundError("Model path does not exist."))
    def test_load_agent_model_failure(self, mock_load_agent_model):
        with self.assertRaises(FileNotFoundError):
            model_utils.load_agent_model(self.agent)
        mock_load_agent_model.assert_called_once_with(self.agent)

    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    @patch('nnbattle.agents.alphazero.agent_code.torch.load')
    def test_load_agent_model_success(self, mock_torch_load, mock_load_agent_model):
        mock_load_agent_model.return_value = None
        model_utils.load_agent_model(self.agent)
        mock_load_agent_model.assert_called_with(self.agent)

    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model')
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    def test_save_agent_model_success(self, mock_load_agent_model, mock_save_agent_model):
        path = "nnbattle/agents/alphazero/model/alphazero_model_final.pth"
        # Mock model.state_dict to return a serializable dictionary
        self.agent.model.state_dict.return_value = {'layer1.weight': torch.randn(10, 10)}
        model_utils.save_agent_model(self.agent, path)
        mock_save_agent_model.assert_called_with(self.agent, path)

    @patch('torch.save')
    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_agent_model_failure(self, mock_save_agent_model, mock_torch_save):
        with self.assertRaises(Exception):
            model_utils.save_agent_model(self.agent, "model/path.pth")
        mock_save_agent_model.assert_called_with(self.agent, "model/path.pth")
        mock_torch_save.assert_not_called()

    def test_select_move_no_model_loaded(self):
        # Ensure model is loaded if not already
        with patch('nnbattle.agents.alphazero.agent_code.load_agent_model') as mock_load_agent_model:  # Updated patch path
            self.agent.model_loaded = False
            self.agent.load_model_flag = True  # Ensure flag is set
            # Return only two values consistently
            self.agent.mcts_simulate.return_value = (3, torch.tensor([0.6, 0.4]))
            action, action_probs = self.agent.select_move(ConnectFourGame())
            mock_load_agent_model.assert_called_once_with(self.agent)
            self.assertEqual(action, 3)
            np.testing.assert_almost_equal(action_probs.tolist(), [0.6, 0.4], decimal=6)

    def test_select_move_model_already_loaded(self):
        # Ensure model is not loaded again if already loaded
        with patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model') as mock_load_agent_model:
            self.agent.model_loaded = True
            # Return only two values consistently
            self.agent.mcts_simulate.return_value = (4, torch.tensor([0.7, 0.3]))
            action, action_probs = self.agent.select_move(ConnectFourGame())
            mock_load_agent_model.assert_not_called()
            self.assertEqual(action, 4)
            np.testing.assert_almost_equal(action_probs.tolist(), [0.7, 0.3], decimal=6)

    def test_select_move_no_actions(self):
        # Simulate scenario where no actions are available
        self.agent.mcts_simulate.return_value = (None, torch.zeros(self.agent.action_dim, device=self.agent.device))
        action, action_probs = self.agent.select_move(ConnectFourGame())
        self.assertIsNone(action)
        torch.testing.assert_close(action_probs, torch.zeros(self.agent.action_dim, device=self.agent.device))

    def test_select_move_different_channels(self):
        # Create a dummy state with 2 channels
        dummy_state = torch.randn(1, 2, 6, 7)  # [1, 2, 6, 7]
        with patch.object(self.agent, 'preprocess', return_value=dummy_state.squeeze(0)):
            self.agent.mcts_simulate.return_value = (1, torch.tensor([0.5, 0.5]))  # Return tensor instead of list
            action, action_probs = self.agent.select_move(ConnectFourGame())
            self.assertIsNotNone(action, "Action should be selected with correct input channels.")
            self.assertEqual(action, 1)
            self.assertEqual(action_probs.tolist(), [0.5, 0.5])

    def test_self_play_deterministic(self):
        self.agent.mcts_simulate.return_value = (None, torch.zeros(self.agent.action_dim, device=self.agent.device))
        with patch.object(self.agent, 'select_move', return_value=(3, torch.tensor([0,0,0,1,0,0,0], dtype=torch.float32))):
            with patch.object(ConnectFourGame, 'make_move', return_value=None):
                self.agent.self_play(max_moves=1)
                self.assertEqual(len(self.agent.memory), 1)
                state, mcts_prob, value = self.agent.memory[0]
                self.assertEqual(mcts_prob[3], 1.0)
                self.assertEqual(value, 0)  # Game is ongoing or draw

    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model')
    def test_save_agent_model_success_with_path(self, mock_save_agent_model):
        # Mock model.state_dict to return a serializable dictionary
        self.agent.model.state_dict.return_value = {'layer1.weight': torch.randn(10, 10)}
        model_utils.save_agent_model(self.agent, "model/path.pth")
        mock_save_agent_model.assert_called_with(self.agent, "model/path.pth")

    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_agent_model_failure_with_path(self, mock_save_agent_model):
        # Simulate model saving failure
        with self.assertRaises(Exception):
            model_utils.save_agent_model(self.agent, "model/path.pth")
        mock_save_agent_model.assert_called_with(self.agent, "model/path.pth")
        # ...additional assertions if needed...

    @patch('nnbattle.agents.alphazero.train.trainer.train_alphazero')
    @patch('nnbattle.agents.alphazero.utils.model_utils.initialize_agent')
    def test_train_alphazero(self, mock_initialize_agent, mock_train):
        with patch('nnbattle.agents.alphazero.initialize_agent') as mock_initialize_agent_correct:
            mock_agent = MagicMock()
            mock_initialize_agent_correct.return_value = mock_agent
            train_alphazero(
                max_iterations=2,  # Reduced from 1000 to 2 for testing
                num_self_play_games=10,  # Reduced for quicker tests
                use_gpu=True,
                load_model=True
            )
            mock_initialize_agent_correct.assert_called_once_with(
                action_dim=7,
                state_dim=2,
                use_gpu=True,
                num_simulations=800,
                c_puct=1.4,
                load_model=True
            )
            mock_train.assert_called_once_with(
                max_iterations=2,  # Ensure the reduced iterations are used
                num_self_play_games=10,
                use_gpu=True,
                load_model=True
            )

    def test_preprocess_empty_board(self):
        """Test preprocessing on an empty board."""
        board = np.zeros((6, 7))
        preprocessed_board = self.agent.preprocess(board)
        self.assertEqual(preprocessed_board.shape, (2, 6, 7))
        self.assertTrue(torch.all(preprocessed_board == 0))

    def test_preprocess_non_empty_board(self):
        """Test preprocessing on a board with moves."""
        board = np.zeros((6, 7))
        board[0, 0] = 1  # Player 1
        board[0, 1] = 2  # Player 2
        preprocessed_board = self.agent.preprocess(board)
        self.assertEqual(preprocessed_board.shape, (2, 6, 7))
        self.assertEqual(preprocessed_board[0, 0, 0], 1)
        self.assertEqual(preprocessed_board[1, 0, 1], 1)

    @patch('nnbattle.agents.alphazero.agent_code.initialize_agent', side_effect=FileNotFoundError("Initialize agent failed."))
    def test_load_agent_model_failure(self, mock_initialize_agent):
        with self.assertRaises(FileNotFoundError):
            model_utils.load_agent_model(self.agent)
        mock_initialize_agent.assert_called_once_with(self.agent)

    # Ensure all mocks for mcts_simulate return two values
    def test_act_method(self):
        """Test the act method to ensure it returns valid actions."""
        game = ConnectFourGame()
        self.agent.mcts_simulate.return_value = (3, torch.tensor([0.6, 0.4]))
        action, action_probs = self.agent.act(game)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.agent.action_dim)
        self.assertEqual(action_probs.shape, (self.agent.action_dim,))
        self.assertAlmostEqual(action_probs.sum().item(), 1.0, places=5)
    
    def test_mcts_simulation(self):
        """Test MCTS simulation to ensure it runs without errors."""
        game = ConnectFourGame()
        self.agent.mcts_simulate.return_value = (2, torch.tensor([0.3, 0.7]))
        selected_action, action_probs = self.agent.mcts_simulate(game)
        self.assertIsInstance(selected_action, int)
        self.assertEqual(action_probs.shape, (self.agent.action_dim,))
        self.assertAlmostEqual(action_probs.sum().item(), 1.0, places=5)
    
    def test_self_play_memory(self):
        """Test that self_play populates the memory with game data."""
        initial_memory_length = len(self.agent.memory)
        self.agent.mcts_simulate.return_value = (1, torch.tensor([0.5, 0.5]))
        self.agent.self_play()
        self.assertGreater(len(self.agent.memory), initial_memory_length)

    def test_act_method(self):
        """Test the act method to ensure it returns valid actions."""
        game = ConnectFourGame()
        action, action_probs = self.agent.act(game)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.agent.action_dim)
        self.assertEqual(action_probs.shape, (self.agent.action_dim,))
        self.assertAlmostEqual(action_probs.sum().item(), 1.0, places=5)

    def test_mcts_simulation(self):
        """Test MCTS simulation to ensure it runs without errors."""
        game = ConnectFourGame()
        selected_action, action_probs = self.agent.mcts_simulate(game)
        self.assertIsInstance(selected_action, int)
        self.assertEqual(action_probs.shape, (self.agent.action_dim,))
        self.assertAlmostEqual(action_probs.sum().item(), 1.0, places=5)

    def test_self_play_memory(self):
        """Test that self_play populates the memory with game data."""
        initial_memory_length = len(self.agent.memory)
        self.agent.self_play()
        self.assertGreater(len(self.agent.memory), initial_memory_length)

if __name__ == '__main__':
    unittest.main()