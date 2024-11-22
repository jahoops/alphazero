# tests/test_train_alpha_zero.py

import unittest
from unittest.mock import MagicMock, patch

from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent
from nnbattle.agents.alphazero.train.train_alpha_zero import train_alphazero


class TestTrainAlphaZero(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock(spec=AlphaZeroAgent)
    
    @patch('nnbattle.agents.train.train_alpha_zero.logger')
    def test_initialize_agent(self, mock_logger):
        state_dim = 42
        action_dim = 7
        agent = AlphaZeroAgent.initialize_agent()
        self.assertIsInstance(agent, AlphaZeroAgent)
        mock_logger.info.assert_called_with("Agent initialized successfully.")
    
    @patch('nnbattle.agents.train.train_alpha_zero.logger')
    def test_load_agent_model_success(self, mock_logger):
        self.load_agent_model(self.agent, "model/path.pth")
        AlphaZeroAgent.load_model.assert_called_with("model/path.pth")
        mock_logger.info.assert_called_with("Model loaded from model/path.pth")
        mock_logger.info.assert_called_with("Model loaded successfully from model/path.pth")
        self.agent.load_model.assert_called_with("model/path.pth")
    
    @patch('nnbattle.agents.train.train_alpha_zero.logger')
    def test_load_agent_model_failure(self, mock_logger):
        self.agent.load_model.side_effect = Exception("Load failed")
        with self.assertRaises(Exception):
            self.agent.load_model("model/path.pth")
        self.load_agent_model(self.agent, "model/path.pth")
        self.agent.load_model.assert_called_with("model/path.pth")
        mock_logger.error.assert_called_with("Failed to load model: Load failed")
        mock_logger.error.assert_called_with("Failed to load model: Load failed")
    
    @patch('nnbattle.agents.train.train_alpha_zero.logger')
    def test_save_agent_model_success(self, mock_logger):
        self.save_agent_model(self.agent, "model/path.pth")
        AlphaZeroAgent.save_model.assert_called_with("model/path.pth")
        mock_logger.info.assert_called_with("Model saved to model/path.pth")
        mock_logger.info.assert_called_with("Model saved successfully to model/path.pth")
        self.agent.save_model.assert_called_with("model/path.pth")
    
    @patch('nnbattle.agents.train.train_alpha_zero.logger')
    def test_save_agent_model_failure(self, mock_logger):
        self.agent.save_model.side_effect = Exception("Save failed")
        self.agent.save_model("model/path.pth")
        self.agent.save_model.assert_called_with("model/path.pth")
        mock_logger.error.assert_called_with("Failed to save model: Save failed")
        mock_logger.error.assert_called_with("Failed to save model: Save failed")
    
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.logger')
    @patch('nnbattle.agents.alphazero.agent_code.time')
    def test_perform_training(self, mock_time, mock_logger):
        mock_time.time.side_effect = [0, 100, 200]  # Simulate time progression
        self.agent.perform_training(time_limit=150)
        self.agent.self_play.assert_called()
        self.agent.train_step.assert_called_with(batch_size=32)
        mock_logger.info.assert_any_call("Elapsed time: 100.00s | Memory Size: 0")
        # Since time_limit=150, the loop should run twice (0-100 and 100-200 >150)
        self.agent.perform_training.assert_called_with(time_limit=150)
    
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.logger')
    @patch('nnbattle.agents.alphazero.agent_code.time')
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.initialize_agent')
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.load_model')
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.perform_training')
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.save_model')
    @patch('nnbattle.agents.alphazero.agent_code.AlphaZeroAgent.perform_training')
    def test_train_alphazero(self, mock_save, mock_train, mock_load, mock_initialize, mock_time, mock_logger):
        mock_agent = MagicMock(spec=AlphaZeroAgent)
        mock_initialize.return_value = mock_agent
        AlphaZeroAgent.train_alphazero(time_limit=3600, load_model=True, model_path="model/path.pth")
        mock_initialize.assert_called_with(state_dim=42, action_dim=7, use_gpu=False)
        mock_load.assert_called_with(mock_agent, "model/path.pth")
        mock_train.assert_called_with(mock_agent, 3600)
        mock_save.assert_called_with(mock_agent, "model/path.pth")

if __name__ == '__main__':
    unittest.main()