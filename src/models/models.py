import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import *
from src.utility.geometric import Sphere
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

class BLogisticRegression(nn.Module): 
    def __init__(self, **kwargs):
        super(BLogisticRegression, self).__init__()
        self.linear = nn.Linear(kwargs["input_dim"], 1) # binary classification

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
    def linearize(self, data):
        """
        Computes the first-order Taylor expansion of the LogisticRegression for each element in a batch.
        For a linear model, the output is already the linearized version.

        Args:
            data (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the LogisticRegression for the batch, shape [batch_size, 1].
                - 'gradient': The model's parameters (weights), shape [1, input_dim].
                - 'linearized': The model itself, represented by the weights and bias.
        """
        # Compute the forward pass
        output = self.forward(data)  # Shape: [batch_size, 1]

        # Extract the model's parameters (weights and bias)
        weights = self.linear.weight.detach()  # Shape: [1, input_dim]
        #bias = self.linear.bias.detach()  # Shape: [1]

        return {
            "output": output.detach(),  # Raw model outputs
            "gradient": weights.detach(),  # Model's gradients w.r.t input are the weights
            "linearized": self,  # Linearized representation of the model
        }
        #TODO define a linearize function that linearize the metod at some datapoints, since LogisticRegression is linear, it doesn't do anything but agrees to the same output
        #       of MLP.linearize

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_dropout = kwargs["dropout"] > 0.0
        self.apply_softmax = kwargs.get("apply_softmax", False)  # Optional parameter
        
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = kwargs["input_dim"]
        for hidden_dim in kwargs["hidden_layers"]:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_dropout:
                self.layers.append(nn.Dropout(kwargs["dropout"]))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, kwargs["nclasses"]))

    def forward(self, x: torch.Tensor):
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = F.relu(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        # Apply softmax if required
        if self.apply_softmax:
            x = F.softmax(x, dim=-1)
        
        return x

    def linearize(self, x: torch.Tensor):
        """
        Computes the first-order Taylor expansion of the MLP for each element in a batch.

        Args:
            x (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the MLP for the batch, shape [batch_size, nclasses].
                - 'gradient': The gradient of the output w.r.t. the input, shape [batch_size, nclasses, input_dim].
                - 'linearized': The linearized approximation for each input in the batch, shape [batch_size, nclasses].
        """
        # Ensure gradients are tracked for the input
        x.requires_grad_(True)

        # Compute the forward pass and optionally apply softmax
        def forward_single_input(x_single):
            logits = self.forward(x_single.unsqueeze(0)).squeeze(0)
            if self.apply_softmax:
                return F.softmax(logits, dim=-1)  # Apply softmax to logits
            return logits  # Raw logits if softmax is not applied

        # Compute the Jacobian for each input in the batch
        # Shape of jacobian: [batch_size, nclasses, input_dim]
        jacobian = torch.autograd.functional.jacobian(
            forward_single_input, x, create_graph=True
        )

        # Compute the forward pass
        output = self.forward(x)  # Shape: [batch_size, nclasses]

        # Apply softmax to the output if needed
        if self.apply_softmax:
            output = F.softmax(output, dim=-1)

        # Compute the linearized approximation
        # Delta x: perturbation around the input
        delta_x = x - x.detach()
        linearized_approximation = output + torch.einsum("bki,bi->bk", jacobian, delta_x)   #TODO this line contains an error

        return {
            "output": output.detach(),  # Original outputs
            "gradient": jacobian.detach(),  # Gradients for each input-output pair
            "linearized": linearized_approximation.detach(),  # First-order approximation
        }

def extract_embeddings_hook(module, input, output):
    
    module.embeddings = output
    
    
class CNN(nn.Module):
    
    def __init__(self, dimension_input: int, classes: int, channel_input: int, channel_list: list[int], kernel_list: list[int]):
        super(CNN, self).__init__()
        self.shapes: list[int] = []
        self.layers: nn.ModuleList = nn.ModuleList()
        current_channel = channel_input
        
        for channels, kernel in zip(channel_list, kernel_list):
            self.layers.append(nn.Conv2d(in_channels=current_channel, out_channels=channels, kernel_size=kernel))
            self.shapes.append(self.output_shape(edge=dimension_input, kernel_size=kernel))
            dimension_input = self.shapes[-1]
            self.layers.append(nn.MaxPool2d(kernel_size=2))
            self.shapes.append(self.output_shape(edge=dimension_input, kernel_size=2, stride=2))
            current_channel = channels
            dimension_input = self.shapes[-1]
            
        self.layers.append(nn.Linear(channel_list[-1]*self.shapes[-1]*self.shapes[-1], classes))


    def forward(self, x: torch.Tensor):
        
        for layer in self.layers[:-1]:
            x = layer(x)
            
            if isinstance(layer, nn.MaxPool2d):
                x = F.relu(x)
                
        x = torch.flatten(x, start_dim=1)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=-1)
    
    def output_shape(self, edge: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
        
        out = (edge + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return out
    


class NoiseModule(nn.Module):
    
    def __init__(self, 
                 shape: Tuple[int, ...],
                 distribution: str = "normal", 
                 n_samples: int = 10, 
                 radius: float = 1.0,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.__sphere: Sphere = Sphere
        self.__random_function = self.__sphere.random_normal_points_in_sphere if distribution == "normal" else self.__sphere.random_uniform_points_in_sphere
        self.__perturbation = self.__random_function(num_points=n_samples, shape=shape, radius=radius)
        self.__n_samples = n_samples
        self.__shape = shape
        
    def forward(self, x: Tensor):
        
        batch_size: int = x.shape[0]
        unit_dims: Tuple[int, ...] = (1, ) 
        new_shape: Tuple[int, ...] = (self.__n_samples, *unit_dims, *self.__shape)
        perturbation: Tensor = self.__perturbation.view(new_shape)
        repeat_dims: Tuple[int, ...] = (1, batch_size, *((1, )*len(new_shape[2:])))
        perturbation: Tensor = perturbation.repeat(repeat_dims)       
        data: Tensor = data.to(device=self.device) 
        sample_perturbed: Tensor = x + perturbation 
        batch_dims: Tuple[int, ...] = (-1, *new_shape[2:])
        sample_perturbed: Tensor = sample_perturbed.reshape(batch_dims)
        
        return sample_perturbed
        
    