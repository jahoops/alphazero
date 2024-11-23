
import unittest
import torch
from nnbattle.agents.alphazero.lightning_module import ConnectFourLightningModule
from nnbattle.agents.alphazero.agent_code import initialize_agent

class TestLightningModule(unittest.TestCase):
    def setUp(self):
        self.agent = initialize_agent(load_model=False)
        self.lightning_module = ConnectFourLightningModule(self.agent)

    def test_configure_optimizers(self):
        """Test that the optimizer is correctly configured."""
        optimizer = self.lightning_module.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_training_step_loss(self):
        """Test that the training step returns a valid loss."""
        batch = (
            torch.randn(4, 2, 6, 7),
            torch.softmax(torch.randn(4, self.agent.action_dim), dim=1),
            torch.randn(4)
        )
        loss = self.lightning_module.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))

    def test_loss_function(self):
        """Test the custom loss function."""
        outputs = (
            torch.randn(4, self.agent.action_dim),
            torch.randn(4, 1)
        )
        targets_policy = torch.softmax(torch.randn(4, self.agent.action_dim), dim=1)
        targets_value = torch.randn(4)
        loss = self.lightning_module.loss_function(outputs, targets_policy, targets_value)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main()