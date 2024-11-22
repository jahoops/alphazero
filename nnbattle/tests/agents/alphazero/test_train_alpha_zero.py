# tests/agents/alphazero/test_train_alpha_zero.py

import unittest
from unittest.mock import MagicMock, patch
import torch

from nnbattle.agents.alphazero.utils.model_utils import (
    load_agent_model, 
    save_agent_model, 
    MODEL_PATH
)
from nnbattle.agents.alphazero.train.trainer import train_alphazero
from nnbattle.agents.alphazero.agent_code import initialize_agent

class TestTrainAlphaZero(unittest.TestCase):
    def setUp(self):
        # Update patch path to correct location
        with patch('nnbattle.agents.alphazero.agent_code.initialize_agent') as mock_initialize:
            mock_agent = MagicMock()
            # Ensure that model.state_dict returns a realistic dictionary
            mock_agent.model.state_dict.return_value = {
                'conv1.weight': torch.randn(16, 2, 3, 3),
                'conv1.bias': torch.randn(16),
                'conv2.weight': torch.randn(32, 16, 3, 3),
                'conv2.bias': torch.randn(32),
                'fc1.weight': torch.randn(128, 32 * 6 * 7),
                'fc1.bias': torch.randn(128),
                'fc2.weight': torch.randn(7, 128),
                'fc2.bias': torch.randn(7)
            }
            mock_initialize.return_value = mock_agent
            # Set a real model_path to avoid undefined attribute issues
            mock_agent.model_path = MODEL_PATH
            self.agent = initialize_agent(
                state_dim=2,
                action_dim=7,
                use_gpu=False,
                num_simulations=800,
                c_puct=1.4,
                load_model=False
            )

    # Update all patch paths from utils.model_utils.initialize_agent to agent_code.initialize_agent
    @patch('nnbattle.agents.alphazero.agent_code.initialize_agent')
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model')
    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    def test_train_alphazero_success(
        self, mock_logger, mock_save_agent_model, mock_load_agent_model, mock_initialize_agent
    ):
        # Mocking initialize_agent to return a mock agent
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        # Mock other dependencies used within train_alphazero
        with patch('nnbattle.agents.alphazero.train.trainer.self_play') as mock_self_play:
            mock_self_play.return_value = []

            with patch('nnbattle.agents.alphazero.train.trainer.Connect4LightningModule'):
                with patch('nnbattle.agents.alphazero.train.trainer.pl.Trainer') as mock_trainer_class:
                    mock_trainer = MagicMock()
                    mock_trainer_class.return_value = mock_trainer

                    # Call the function under test
                    train_alphazero(
                        time_limit=1,  # Use small time for testing
                        num_self_play_games=1,
                        use_gpu=False,
                        load_model=False
                    )

                    # Assertions
                    mock_initialize_agent.assert_called_once_with(
                        action_dim=7,
                        state_dim=2,
                        use_gpu=False,
                        num_simulations=800,
                        c_puct=1.4,
                        load_model=False
                    )
                    mock_load_agent_model.assert_called_once_with(mock_agent)
                    mock_self_play.assert_called_once_with(mock_agent, 1)
                    mock_trainer.fit.assert_called_once()

    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    @patch('nnbattle.agents.alphazero.agent_code.initialize_agent')
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model', side_effect=FileNotFoundError("Model path does not exist."))
    def test_train_alphazero_load_failure(self, mock_load_agent_model, mock_initialize_agent, mock_logger):
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        with self.assertRaises(FileNotFoundError):
            train_alphazero(
                time_limit=3600,
                num_self_play_games=1000,
                use_gpu=True,
                load_model=True
            )
        mock_load_agent_model.assert_called_once_with(mock_agent)
        mock_logger.error.assert_called_with("Model path nnbattle/agents/alphazero/model/alphazero_model_final.pth does not exist.")

    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    @patch('nnbattle.agents.alphazero.agent_code.initialize_agent')
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_train_alphazero_save_failure(
        self, mock_logger, mock_save_agent_model, mock_load_agent_model, mock_initialize_agent
    ):
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        with patch('nnbattle.agents.alphazero.train.trainer.self_play') as mock_self_play:
            mock_self_play.return_value = []

            with patch('nnbattle.agents.alphazero.train.trainer.ConnectFourDataModule'):
                with patch('nnbattle.agents.alphazero.train.trainer.Connect4LightningModule'):
                    with patch('nnbattle.agents.alphazero.train.trainer.pl.Trainer') as mock_trainer_class:
                        mock_trainer = MagicMock()
                        mock_trainer_class.return_value = mock_trainer

                        # Call the function under test
                        with self.assertRaises(Exception):
                            train_alphazero(
                                time_limit=1,
                                num_self_play_games=1,
                                use_gpu=False,
                                load_model=False
                            )

                        mock_save_agent_model.assert_called_once_with(mock_agent, "nnbattle/agents/alphazero/model/alphazero_model_final.pth")
                        mock_logger.error.assert_called_with("Error saving model: Save failed")

    @patch('nnbattle.agents.alphazero.train.trainer.train_alphazero')
    def test_perform_training(self, mock_train):
        self.agent.perform_training = MagicMock()
        self.agent.perform_training()
        mock_train.assert_called_with(
            time_limit=3600,
            num_self_play_games=1000,
            use_gpu=self.agent.device.type == 'cuda',
            load_model=self.agent.model_loaded
        )

    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    @patch('nnbattle.agents.alphazero.train.trainer.train_alphazero')
    def test_train_alphazero(self, mock_train, mock_load_agent_model):
        with patch('nnbattle.agents.alphazero.agent_code.initialize_agent') as mock_initialize:
            mock_initialize.return_value = self.agent
            with patch('nnbattle.agents.alphazero.train.trainer.self_play') as mock_self_play, \
                 patch('nnbattle.agents.alphazero.train.trainer.ConnectFourDataModule'), \
                 patch('nnbattle.agents.alphazero.train.trainer.Connect4LightningModule'), \
                 patch('nnbattle.agents.alphazero.train.trainer.pl.Trainer') as mock_trainer_class, \
                 patch('nnbattle.agents.alphazero.train.trainer.save_agent_model') as mock_save_agent_model:
                     
                mock_agent = MagicMock()
                mock_initialize.return_value = mock_agent
                mock_self_play.return_value = []
                mock_trainer = MagicMock()
                mock_trainer_class.return_value = mock_trainer

                train_alphazero(
                    time_limit=1,
                    num_self_play_games=1,
                    use_gpu=False,
                    load_model=False
                )

                mock_initialize.assert_called_once_with(
                    action_dim=7,
                    state_dim=2,
                    use_gpu=False,
                    num_simulations=800,
                    c_puct=1.4,
                    load_model=False
                )
                mock_load_agent_model.assert_called_once_with(mock_agent)
                mock_self_play.assert_called_once_with(mock_agent, 1)
                mock_trainer.fit.assert_called_once()
                mock_save_agent_model.assert_called_once_with(mock_agent, "nnbattle/agents/alphazero/model/alphazero_model_final.pth")

    @patch('nnbattle.agents.alphazero.train.trainer.initialize_agent')  # Updated patch path
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model')
    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    def test_train_alphazero_success(
        self, mock_logger, mock_save_agent_model, mock_load_agent_model, mock_initialize_agent
    ):
        # Mocking initialize_agent to return a mock agent
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        # Mock other dependencies used within train_alphazero
        with patch('nnbattle.agents.alphazero.train.trainer.self_play') as mock_self_play:
            mock_self_play.return_value = []

            with patch('nnbattle.agents.alphazero.train.trainer.Connect4LightningModule'):
                with patch('nnbattle.agents.alphazero.train.trainer.pl.Trainer') as mock_trainer_class:
                    mock_trainer = MagicMock()
                    mock_trainer_class.return_value = mock_trainer

                    # Call the function under test
                    train_alphazero(
                        time_limit=1,  # Use small time for testing
                        num_self_play_games=1,
                        use_gpu=False,
                        load_model=False
                    )

                    # Assertions
                    mock_initialize_agent.assert_called_once_with(
                        action_dim=7,
                        state_dim=2,
                        use_gpu=False,
                        num_simulations=800,
                        c_puct=1.4,
                        load_model=False
                    )
                    mock_load_agent_model.assert_called_once_with(mock_agent)
                    mock_self_play.assert_called_once_with(mock_agent, 1)
                    mock_trainer.fit.assert_called_once()

    @patch('nnbattle.agents.alphazero.agent_code.initialize_agent')
    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    def test_initialize_agent(self, mock_logger, mock_initialize_agent):
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent
        agent = initialize_agent()
        self.assertIsInstance(agent, MagicMock)
        mock_logger.info.assert_called_with("Agent initialized successfully.")

    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model')
    def test_load_agent_model_success(self, mock_load_agent_model, mock_logger):
        load_agent_model(self.agent)
        mock_load_agent_model.assert_called_with(self.agent)
        mock_logger.info.assert_called_with("Model loaded successfully from nnbattle/agents/alphazero/model/alphazero_model_final.pth")

    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    @patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model', side_effect=FileNotFoundError("Model path does not exist."))
    def test_load_agent_model_failure(self, mock_load_agent_model, mock_logger):
        with self.assertRaises(FileNotFoundError):
            load_agent_model(self.agent)
        mock_logger.error.assert_called_with("Model path nnbattle/agents/alphazero/model/alphazero_model_final.pth does not exist.")

    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model')
    def test_save_agent_model_success(self, mock_save_agent_model, mock_logger):
        save_agent_model(self.agent)
        mock_save_agent_model.assert_called_with(self.agent, "nnbattle/agents/alphazero/model/alphazero_model_final.pth")
        mock_logger.info.assert_called_with("Model saved to nnbattle/agents/alphazero/model/alphazero_model_final.pth.")

    @patch('nnbattle.agents.alphazero.train.trainer.logger')
    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_agent_model_failure(self, mock_save_agent_model, mock_logger):
        with self.assertRaises(Exception):
            save_agent_model(self.agent)
        mock_logger.error.assert_called_with("Error saving model: Save failed")

    def test_select_move_no_model_loaded(self):
        # Ensure model is loaded if not already
        with patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model') as mock_load_agent_model:
            self.agent.model_loaded = False
            self.agent.mcts_simulate.return_value = (3, [3,4], [0.6, 0.4])
            action, action_probs = self.agent.select_move(ConnectFourGame())
            mock_load_agent_model.assert_called_once_with(self.agent)
            self.assertEqual(action, 3)

    def test_select_move_model_already_loaded(self):
        # Ensure model is not loaded again if already loaded
        with patch('nnbattle.agents.alphazero.utils.model_utils.load_agent_model') as mock_load_agent_model:
            self.agent.model_loaded = True
            self.agent.mcts_simulate.return_value = (4, [2,5], [0.7, 0.3])
            action, action_probs = self.agent.select_move(ConnectFourGame())
            mock_load_agent_model.assert_not_called()
            self.assertEqual(action, 4)

    def test_select_move_no_actions(self):
        # Simulate scenario where no actions are available
        self.agent.mcts_simulate.return_value = (None, [], [])
        action, action_probs = self.agent.select_move(ConnectFourGame())
        self.assertIsNone(action)

    def test_select_move_different_channels(self):
        # Create a dummy state with 2 channels
        dummy_state = torch.randn(1, 2, 6, 7)  # [1, 2, 6, 7]
        with patch.object(self.agent, 'preprocess', return_value=dummy_state.squeeze(0)):
            action, action_probs = self.agent.select_move(ConnectFourGame())
            self.assertIsNotNone(action, "Action should be selected with correct input channels.")

    def test_self_play_deterministic(self):
        self.agent.mcts_simulate.return_value = (None, [], [])
        with patch.object(self.agent, 'select_move', return_value=(3, torch.tensor([0,0,0,1,0,0,0], dtype=torch.float32))):
            with patch.object(ConnectFourGame, 'make_move', return_value=None):
                self.agent.self_play()
                self.assertEqual(len(self.agent.memory), 1)
                state, mcts_prob, value = self.agent.memory[0]
                self.assertEqual(mcts_prob[3], 1.0)
                self.assertEqual(value, -1)  # Assuming player 1 lost

    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model')
    def test_save_model_success_with_path(self, mock_save_agent_model):
        # Simulate successful model saving
        self.agent.model_path = MODEL_PATH  # Ensure model_path is set
        save_agent_model(self.agent, MODEL_PATH)
        mock_save_agent_model.assert_called_with(self.agent, MODEL_PATH)

    @patch('nnbattle.agents.alphazero.utils.model_utils.save_agent_model', side_effect=Exception("Save failed"))
    def test_save_model_failure_with_path(self, mock_save_agent_model):
        # Simulate model saving failure
        self.agent.model_path = MODEL_PATH  # Ensure model_path is set
        with self.assertRaises(Exception):
            save_agent_model(self.agent, MODEL_PATH)
        # ...additional assertions if needed...

    if __name__ == '__main__':
        unittest.main()