# tests/test_train_alpha_zero.py

import unittest
from unittest.mock import MagicMock, patch

from nnbattle.agents.alphazero.utils import initialize_agent, load_agent_model, save_agent_model  # Updated imports
from nnbattle.agents.alphazero.train.train_alpha_zero import train_alphazero

class TestTrainAlphaZero(unittest.TestCase):
    def setUp(self):
        # Initialize agent without loading the model to avoid side effects
        with patch('nnbattle.agents.alphazero.utils.initialize_agent') as mock_initialize:
            mock_agent = MagicMock()
            mock_initialize.return_value = mock_agent
            self.agent = initialize_agent(
                state_dim=2,
                action_dim=7,
                use_gpu=False,
                # model_path="model/path.pth",  # Removed to use MODEL_PATH from utils.py
                num_simulations=800,
                c_puct=1.4,
                load_model=False  # Set to False to avoid loading during tests
            )

    @patch('nnbattle.agents.alphazero.utils.initialize_agent')  # Patch initialize_agent
    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.logger')
    def test_initialize_agent(self, mock_logger, mock_initialize_agent):
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent
        agent = initialize_agent()
        self.assertIsInstance(agent, MagicMock)  # Since agent is mocked in tests
        mock_logger.info.assert_called_with("Agent initialized successfully.")

    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.logger')
    @patch('nnbattle.agents.alphazero.utils.load_agent_model')  # Patch load_agent_model
    def test_load_agent_model_success(self, mock_load_agent_model, mock_logger):
        mock_load_agent_model.return_value = None  # Assuming load_agent_model doesn't return anything
        load_agent_model(self.agent)
        mock_load_agent_model.assert_called_with(self.agent)
        mock_logger.info.assert_called_with("Model loaded successfully from nnbattle/agents/alphazero/model/alphazero_model_final.pth")

    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.logger')
    @patch('nnbattle.agents.alphazero.utils.load_agent_model', side_effect=FileNotFoundError("Model path does not exist."))
    def test_load_agent_model_failure(self, mock_load_agent_model, mock_logger):
        with self.assertRaises(FileNotFoundError):
            load_agent_model(self.agent)
        mock_logger.error.assert_called_with("Model path nnbattle/agents/alphazero/model/alphazero_model_final.pth does not exist.")

    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.logger')
    @patch('nnbattle.agents.alphazero.utils.save_agent_model')  # Patch save_agent_model
    def test_save_agent_model_success(self, mock_save_agent_model, mock_logger):
        save_agent_model(self.agent)  # Removed the path parameter
        mock_save_agent_model.assert_called_with(self.agent)
        mock_logger.info.assert_called_with("Model saved to nnbattle/agents/alphazero/model/alphazero_model_final.pth.")

    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.logger')
    @patch('nnbattle.agents.alphazero.utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_agent_model_failure(self, mock_save_agent_model, mock_logger):
        with self.assertRaises(Exception):
            save_agent_model(self.agent)  # Removed the path parameter
        mock_logger.error.assert_called_with("Error saving model: Save failed")

    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.train_alphazero')
    def test_perform_training(self, mock_train):
        self.agent.perform_training = MagicMock()  # Mock perform_training if it's a method
        self.agent.perform_training()
        mock_train.assert_called_with(
            time_limit=3600,
            num_self_play_games=1000,
            use_gpu=self.agent.device.type == 'cuda',
            load_model=self.agent.model_loaded
        )

    @patch('nnbattle.agents.alphazero.utils.initialize_agent')  # Patch initialize_agent
    @patch('nnbattle.agents.alphazero.train.train_alpha_zero.train_alphazero')
    def test_train_alphazero(self, mock_train, mock_initialize_agent):
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent
        train_alphazero(
            time_limit=3600,
            num_self_play_games=1000,
            use_gpu=True,
            load_model=True
        )
        mock_initialize_agent.assert_called_once_with(
            action_dim=7,
            state_dim=2,
            use_gpu=True,
            # model_path="nnbattle/agents/alphazero/model/alphazero_model_final.pth",  # Removed to use default
            num_simulations=800,
            c_puct=1.4,
            load_model=True
        )
        mock_train.assert_called_once_with(
            time_limit=3600,
            num_self_play_games=1000,
            use_gpu=True,
            load_model=True
        )

if __name__ == '__main__':
    unittest.main()