import torch
from torch import nn
import torch.nn.functional as F
seed = 42

torch.manual_seed(seed)

# Additional steps to enforce determinism
# Note: These settings can degrade performance and may not guarantee complete reproducibility across different PyTorch releases or different platforms like CPUs and GPUs.

# Ensuring that all operations are deterministic on GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        self.fc1 = nn.Linear(96, 15)
        self.fc2 = nn.Linear(15, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 96)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)