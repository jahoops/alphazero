import torch
from nnbattle.agents.alphazero.network import Connect4Net

# Define the model architecture
model = Connect4Net(state_dim=2, action_dim=7)

# Save the model state_dict
MODEL_PATH = "nnbattle/agents/alphazero/model/alphazero_model_final.pth"
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully.")