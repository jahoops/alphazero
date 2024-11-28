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
        self.automatic_optimization = False  # Manual optimization for speed
        self._train_dataloader = None
        self._val_dataloader = None
        
        # Disable validation by default for speed
        self.should_validate = False
        
        # Add performance configurations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.epoch_losses = []
        self.last_gpu_log = 0

    def on_fit_start(self):
        """Called when fit begins."""
        if self.trainer.datamodule is not None:
            self._train_dataloader = self.trainer.datamodule.train_dataloader()
            self._val_dataloader = self.trainer.datamodule.val_dataloader()

    def forward(self, x):
        # Ensure x has shape [batch_size, 3, 6, 7]
        assert x.shape[1:] == (3, 6, 7), f"Input tensor has incorrect shape: {x.shape}"
        # Use self.model instead of self.agent.model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Handle training step with improved loss calculation."""
        opt = self.optimizers()
        
        # Zero gradients and compute loss
        opt.zero_grad()
        states, mcts_probs, rewards = batch
        logits, values = self(states)
        
        # Calculate losses with adjusted weights
        value_loss = F.mse_loss(values.squeeze(-1), rewards)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
        loss = 0.5 * value_loss + policy_loss  # Adjusted loss weights
        
        # Manual backward and optimization
        self.manual_backward(loss)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        opt.step()
        self.last_loss = loss.item()
        
        # Store loss for epoch averaging
        self.epoch_losses.append(loss.item())
        return {'loss': loss.item()}

    def on_train_epoch_end(self):
        """Log epoch-level metrics and GPU stats."""
        epoch = self.current_epoch
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0
        
        logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        self.log_gpu_stats()
        
        # Reset epoch losses
        self.epoch_losses = []
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get metrics that were logged during training steps
        metrics = self.trainer.callback_metrics
        
        # Log epoch-level metrics if they exist
        if 'loss' in metrics:
            self.log('train_loss_epoch', metrics['loss'], on_epoch=True, prog_bar=True)
        if 'train_loss' in metrics:
            self.log('train_loss_epoch', metrics['train_loss'], on_epoch=True, prog_bar=True)

    def log_gpu_stats(self):
        """Log GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"GPU Memory [GB] - Allocated: {allocated:.2f}, Reserved: {reserved:.2f}, Peak: {max_memory:.2f}")

    def configure_optimizers(self):
        """Improved optimizer configuration."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4  # Add L2 regularization
        )
        return optimizer

    def loss_function(self, outputs, targets_policy, targets_value):
        logits, values = outputs
        value_loss = F.mse_loss(values.squeeze(), targets_value)
        policy_loss = -torch.mean(torch.sum(targets_policy * F.log_softmax(logits, dim=1), dim=1))
        return value_loss + policy_loss

    def validation_step(self, batch, batch_idx):
        """Handle validation with MCTS-generated data."""
        if self._val_dataloader is None:
            self._val_dataloader = self.trainer.val_dataloader
            
        logger.debug(f"Validation Step - Processing MCTS batch {batch_idx}")
        
        # Store original mode
        was_training = self.model.training
        
        # Set eval mode for validation
        self.model.eval()

        try:
            # Process MCTS validation data
            with torch.no_grad():
                states, mcts_probs, rewards = batch
                logits, values = self.forward(states)
                values = values.squeeze(-1)
                
                # Calculate validation metrics from MCTS data
                value_loss = F.mse_loss(values, rewards)
                policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
                total_loss = value_loss + policy_loss
                
                self.log('val_mcts_value_loss', value_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_mcts_policy_loss', policy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
                
        finally:
            # Restore original mode
            self.model.train(was_training)

    # Remove unnecessary hooks and callbacks
    def on_train_start(self): pass
    def on_train_end(self): pass
    def on_validation_start(self): pass
    def on_validation_end(self): pass

# Ensure __all__ is defined for easier imports
__all__ = ['ConnectFourLightningModule']