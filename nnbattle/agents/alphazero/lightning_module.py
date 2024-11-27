# /lightning_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nnbattle.agents.alphazero.network import Connect4Net  # Ensure correct import if needed
import logging

logger = logging.getLogger(__name__)

class ConnectFourLightningModule(pl.LightningModule):
    def __init__(self, agent):
        super().__init__()
        # Register the model as a submodule
        self.model = agent.model
        self.loss_fn = self.loss_function
        self.save_hyperparameters()
        self.automatic_optimization = True

    def forward(self, x):
        # Ensure x has shape [batch_size, 3, 6, 7]
        assert x.shape[1:] == (3, 6, 7), f"Input tensor has incorrect shape: {x.shape}"
        # Use self.model instead of self.agent.model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logger.debug(f"Training Step - Model training mode: {self.model.training}")
        self.model.train()  # Ensure the model is in training mode
        states, mcts_probs, rewards = batch
        # Move tensors to device here after pin_memory has been called
        states = states.to(self.device, non_blocking=True)
        mcts_probs = mcts_probs.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True).view(-1)
        
        # Forward pass
        logits, values = self(states)
        
        # Ensure proper shapes
        values = values.squeeze(-1)
        rewards = rewards.float()
        
        # Calculate losses with added small epsilon to prevent log(0)
        value_loss = F.mse_loss(values, rewards)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
        
        loss = value_loss + policy_loss
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # If you prefer to monitor 'train_loss', add the following:
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # Get metrics that were logged during training steps
        metrics = self.trainer.callback_metrics
        
        # Log epoch-level metrics if they exist
        if 'loss' in metrics:
            self.log('train_loss_epoch', metrics['loss'], on_epoch=True, prog_bar=True)
        if 'train_loss' in metrics:
            self.log('train_loss_epoch', metrics['train_loss'], on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # The optimizer will now include model parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss"  # Ensure this matches the logged metric
        }

    def loss_function(self, outputs, targets_policy, targets_value):
        logits, values = outputs
        value_loss = F.mse_loss(values.squeeze(), targets_value)
        policy_loss = -torch.mean(torch.sum(targets_policy * F.log_softmax(logits, dim=1), dim=1))
        return value_loss + policy_loss

    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation Step - Model training mode: {self.model.training}")
        self.model.eval()  # Ensure the model is in evaluation mode
        """Add a validation step to monitor performance on a separate set."""
        states, mcts_probs, rewards = batch
        # Move tensors to device here after pin_memory has been called
        states = states.to(self.device, non_blocking=True)
        mcts_probs = mcts_probs.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True).view(-1)
        
        logits, values = self.forward(states)
        values = values.squeeze(-1)
        rewards = rewards.float()
        
        value_loss = F.mse_loss(values, rewards)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
        total_loss = value_loss + policy_loss
        
        self.log('validation_value_loss', value_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_policy_loss', policy_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Add detailed logging for validation
        if batch_idx % 10 == 0:
            logger.info(f"Validation Batch {batch_idx}: Value Loss={value_loss.item():.6f}, Policy Loss={policy_loss.item():.6f}, Total Loss={total_loss.item():.6f}")

# Ensure __all__ is defined for easier imports
__all__ = ['ConnectFourLightningModule']