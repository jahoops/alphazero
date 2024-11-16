# alphazero_agent/train/train_alpha_zero.py

import pytorch_lightning as pl
from alphazero_agent.lightning_module import Connect4LightningModule
from alphazero_agent.data_module import ConnectFourDataModule
from alphazero_agent.agent_code import AlphaZeroAgent

def train_alphazero(time_limit, load_model=True, model_path="alphazero_agent/model/alphazero_model_final.pth"):
    # Initialize the agent
    state_dim = 42  # Example: 6 rows * 7 columns
    action_dim = 7  # Number of columns in Connect Four
    agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=False, model_path=model_path)
    
    # Load existing model if required
    if load_model:
        agent.load_model()
    
    # Initialize the LightningModule
    lightning_model = Connect4LightningModule(state_dim=state_dim, action_dim=action_dim, learning_rate=0.001)
    
    # Initialize DataModule
    data_module = ConnectFourDataModule(memory=agent.memory, batch_size=32)
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=20
    )
    
    # Start training
    trainer.fit(lightning_model, datamodule=data_module)
    
    # Save the trained model
    lightning_model.save_hyperparameters()
    trainer.save_checkpoint(model_path)

if __name__ == "__main__":
    # Example: Train for a specific configuration
    train_alphazero(time_limit=3600, load_model=True)