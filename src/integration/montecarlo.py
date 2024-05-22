
from src.functions.abstract_function import GenericFunction
from src.utility.geometric import Sphere
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA
from typing import Tuple, TypeAlias
Tensor: TypeAlias = torch.Tensor


class MontecarloEstimator:

    def __init__(self, 
                 function: torch.nn.Module, 
                 train_set: TensorDataset, 
                 n_samples: int = 1000, 
                 radius: float = 1.0,
                 fraction: float = 0.8,
                 **kwargs) -> None:
        """
        Parameters:
        - function (torch.nn.Module): The neural network model or function to be used.
        - train_set (TensorDataset): The training dataset to be used for training or other operations.
        - shape (Tuple[int, ...]): The shape of the perturbation tensor, if just a dimension use (n, ) otherwise (n, m, ...).
        - n_samples (int, optional): The number of samples to generate. Default is 1000.
        - radius (float, optional): The radius within which to generate the perturbations. Default is 1.0.
        - fraction (float, optional): Fraction used to estimate the counterfactual fraction over the training set. Default is 0.8.
        - **kwargs: Additional keyword arguments that might be required for other operations or configurations.

        Returns:
        - None
        """
        self.sphere: Sphere = Sphere()
        self.function = function.eval()
        self.n_samples = n_samples
        self.random_index = torch.randint(low=0, high=len(train_set), size=(int(len(train_set)*fraction),))
        self.X, _ = train_set[self.random_index]  
        self.shape = self.X[0].shape
        self.perturbation = self.sphere.random_points_in_sphere(num_points=n_samples, shape=self.shape, radius=radius)
        self.volume = self.sphere.hypersphere_volume(dimensions=np.prod(self.shape), radius=radius)
        self.include_volume = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # def compute(self, classifier_out):
    #     """
    #     Perturbation must be of dimension 100, 500, 5 namely P, S, F
    #     Where P is the number of perturbation
    #     S is the number of sample
    #     F is the number of features
    #     Sample must be of dimensio S, F
    #     """ 
    #     sample_perturbed = self.X.to("cuda") + self.perturbation.unsqueeze(1).repeat(1, self.X.shape[0], 1)
    #     out = self.function(sample_perturbed)
    #     out = torch.argmax(out, dim=-1)
    #     classifier_out = classifier_out[self.random_index]
    #     classifier_out = classifier_out.unsqueeze(1)
    #     classifier_out = classifier_out.repeat(1, self.n_samples)
    #     classifier_out = classifier_out.to("cuda")
    #     classifier_out = classifier_out.permute(1, 0)
    #     different = out != classifier_out
    #     average_function_value = torch.sum(different, dim=0)/different.shape[0]
    #     value = self.volume * average_function_value if self.include_volume else average_function_value
        
        
    #     return torch.mean(value).cpu(), torch.std(value).cpu()
    
    def get_counterfactual(self, 
                       X: Tensor, 
                       target: Tensor) -> Tuple[Tensor, Tensor]:
        
        """
        Generate counterfactual samples by perturbing the input tensor `X` and computing the model's output.
        
        The perturbation tensor must have the dimensions (P, S, F), where:
        - P is the number of perturbations.
        - S is the number of samples.
        - F is the number of features.
        
        The input tensor `X` must have dimensions (S, F), where:
        - S is the number of samples.
        - F is the number of features.
        
        Parameters:
        - X (Tensor): The input tensor of shape (batch_size, num_features).
        - target (Tensor): The target tensor, which will be used to generate the counterfactual targets.
        
        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing:
        - out (Tensor): The output tensor from the model after perturbation, reshaped as required.
        - target (Tensor): The repeated and reshaped target tensor to match the perturbation structure.
        """
        batch_size: int = X.shape[0]
        unit_dims: Tuple[int, ...] = (1, ) 
        new_shape: Tuple[int, ...] = (self.n_samples, *unit_dims, *self.shape)
        perturbation: Tensor = self.perturbation.view(new_shape)
        repeat_dims: Tuple[int, ...] = (1, batch_size, *((1, )*len(new_shape[2:])))
        perturbation: Tensor = perturbation.repeat(repeat_dims)       
        X: Tensor = X.to(device=self.device) 
        sample_perturbed: Tensor = X + perturbation 
        batch_dims: Tuple[int, ...] = (-1, *new_shape[2:])
        sample_perturbed: Tensor = sample_perturbed.reshape(batch_dims)
        out: Tensor = self.function(sample_perturbed)
        target: Tensor = torch.argmax(target, dim=-1)
        target: Tensor = target.unsqueeze(1)
        target: Tensor = target.repeat(1, self.n_samples)
        target: Tensor = target.reshape(batch_size * self.n_samples)
        target: Tensor = target.to(self.device)
        
        return out, target
      