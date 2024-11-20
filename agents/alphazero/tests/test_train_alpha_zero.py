# tests/test_train_alpha_zero.py

import unittest
from unittest.mock import MagicMock, patch

from agents.alphazero.agent_code import AlphaZeroAgent
from agents.alphazero.train.train_alpha_zero import (initialize_agent,
                                                     load_agent_model,
                                                     perform_training,
                                                     save_agent_model,
                                                     train_alphazero)


class TestTrainAlphaZero(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock(spec=AlphaZeroAgent)
    
    @patch('agents..train.train_alpha_zero.logger')
    def test_initialize_agent(self, mock_logger):
        state_dim = 42
        action_dim = 7
        agent = initialize_agent(state_dim, action_dim, use_gpu=False)
        self.assertIsInstance(agent, AlphaZeroAgent)
    
    @patch('agents..train.train_alpha_zero.logger')
    def test_load_agent_model_success(self, mock_logger):
        load_agent_model(self.agent, "model/path.pth")
        self.agent.load_model.assert_called_with("model/path.pth")
        mock_logger.info.assert_called_with("Model loaded from model/path.pth")
    
    @patch('agents..train.train_alpha_zero.logger')
    def test_load_agent_model_failure(self, mock_logger):
        self.agent.load_model.side_effect = Exception("Load failed")
        load_agent_model(self.agent, "model/path.pth")
        self.agent.load_model.assert_called_with("model/path.pth")
        mock_logger.error.assert_called_with("Failed to load model: Load failed")
    
    @patch('agents..train.train_alpha_zero.logger')
    def test_save_agent_model_success(self, mock_logger):
        save_agent_model(self.agent, "model/path.pth")
        self.agent.save_model.assert_called_with("model/path.pth")
        mock_logger.info.assert_called_with("Model saved to model/path.pth")
    
    @patch('agents..train.train_alpha_zero.logger')
    def test_save_agent_model_failure(self, mock_logger):
        self.agent.save_model.side_effect = Exception("Save failed")
        save_agent_model(self.agent, "model/path.pth")
        self.agent.save_model.assert_called_with("model/path.pth")
        mock_logger.error.assert_called_with("Failed to save model: Save failed")
    
    @patch('agents..train.train_alpha_zero.logger')
    @patch('agents..train.train_alpha_zero.time')
    def test_perform_training(self, mock_time, mock_logger):
        mock_time.time.side_effect = [0, 100, 200]  # Simulate time progression
        perform_training(self.agent, time_limit=150)
        self.agent.self_play.assert_called()
        self.agent.train_step.assert_called_with(batch_size=32)
        mock_logger.info.assert_any_call("Elapsed time: 100.00s | Memory Size: 0")
        # Since time_limit=150, the loop should run twice (0-100 and 100-200 >150)
    
    @patch('agents..train.train_alpha_zero.initialize_agent')
    @patch('agents..train.train_alpha_zero.load_agent_model')
    @patch('agents..train.train_alpha_zero.perform_training')
    @patch('agents..train.train_alpha_zero.save_agent_model')
    def test_train_alphazero(self, mock_save, mock_train, mock_load, mock_initialize):
        mock_agent = MagicMock(spec=AlphaZeroAgent)
        mock_initialize.return_value = mock_agent
        train_alphazero(time_limit=3600, load_model=True, model_path="model/path.pth")
        mock_initialize.assert_called_with(state_dim=42, action_dim=7, use_gpu=False)
        mock_load.assert_called_with(mock_agent, "model/path.pth")
        mock_train.assert_called_with(mock_agent, 3600)
        mock_save.assert_called_with(mock_agent, "model/path.pth")

if __name__ == '__main__':
    unittest.main()