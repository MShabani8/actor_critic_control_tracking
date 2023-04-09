import torch
import torch.nn as nn
from torch.nn import functional as F

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Actor neural network architecture
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        # Define the layers of the network
        self.output = nn.Sequential(
           nn.Linear(2, hidden_dim),
           nn.Linear(hidden_dim, 1)
        )

    def forward(self, s):
        # Pass the input through the network and return the output
        return self.output(s)

# Define the Value neural network architecture
class ValueNet(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        # Define the layers of the network
        self.output = nn.Sequential(
           nn.Linear(5, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, 1)
        )

    def forward(self, s):
        # Pass the input through the network and return the output
        return self.output(s)

# Initialize the Actor and Value neural networks and send them to the GPU if available
actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)