import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, output_dim))
    
    def forward(self, x):
        # Apply a ReLU activation function to each hidden layer
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # No activation function for the output layer (assuming classification task)
        x = self.layers[-1](x)
        return x

