# alphazero_agent/train/train_alpha_zero.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from alphazero_agent.lightning_module import Connect4LightningModule
from alphazero_agent.data_module import ConnectFourDataModule
from alphazero_agent.agent_code import AlphaZeroAgent
import time

def train_alphazero(time_limit, load_model=True, model_path="alphazero_agent/model/alphazero_model_final.pth"):
    # Initialize the agent
    state_dim = 42  # Example: 6 rows * 7 columns
    action_dim = 7  # Number of columns in Connect Four
    agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=True, model_path=model_path)
    
    # Load existing model if required
    if load_model:
        agent.load_model()
    
    # Initialize the LightningModule
    lightning_model = Connect4LightningModule(state_dim=state_dim, action_dim=action_dim, learning_rate=0.001)
    
    # Initialize DataModule
    data_module = ConnectFourDataModule(agent, batch_size=32)
    
    # Initialize Trainer with checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='alphazero_agent/checkpoints/',
        filename='model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_epochs=1000,  # Set a high number and use time-based stopping
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=20
    )
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # Perform self-play to generate training data
        agent.self_play()
        
        # Update the DataModule's dataset if necessary
        data_module.setup()
        
        # Train for one epoch (or more if desired)
        trainer.fit(lightning_model, datamodule=data_module)
    
    # Save the final model
    trainer.save_checkpoint(model_path)

if __name__ == "__main__":
    # Example: Train for 1 hour (3600 seconds)
    train_alphazero(time_limit=3600, load_model=True)