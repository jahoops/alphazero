# /lightning_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nnbattle.agents.alphazero.network import Connect4Net  # Ensure correct import if needed

class ConnectFourLightningModule(pl.LightningModule):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.loss_fn = self.loss_function

    def forward(self, x):
        return self.agent.model(x)

    def training_step(self, batch, batch_idx):
        states, mcts_probs, rewards = batch
        logits, values = self.forward(states)
        
        # Ensure tensors have requires_grad=True
        if not logits.requires_grad:
            logits = logits.detach().requires_grad_(True)
        if not values.requires_grad:
            values = values.detach().requires_grad_(True)
            
        value_loss = F.mse_loss(values.squeeze(), rewards)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits, dim=1), dim=1))
        loss = value_loss + policy_loss
        
        # Log the individual losses with on_step and on_epoch flags
        self.log('value_loss', value_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('policy_loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        # Add safety check for model parameters
        params = list(self.agent.model.parameters())
        if not params:
            raise ValueError("Model has no parameters to optimize")
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer

    def loss_function(self, outputs, targets_policy, targets_value):
        logits, values = outputs
        value_loss = F.mse_loss(values.squeeze(), targets_value)
        policy_loss = -torch.mean(torch.sum(targets_policy * F.log_softmax(logits, dim=1), dim=1))
        return value_loss + policy_loss

# Ensure __all__ is defined for easier imports
__all__ = ['ConnectFourLightningModule']