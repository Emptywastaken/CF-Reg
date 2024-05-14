
from src.functions.abstract_function import GenericFunction
from src.utility.geometric import random_points_in_n_sphere, hypersphere_volume
import torch
import numpy as np
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class MontecarloEstimator():

    def __init__(self, function, train_set: TensorDataset, n_samples: int = 1000, radius: float = 1.0, **kwargs) -> None:

        self.function = function.eval()
        self.n_samples = n_samples
        self.perturbation = random_points_in_n_sphere(num_points=n_samples, n=9, radius=radius)
        self.random_index = torch.randint(len(train_set), (int(len(train_set)*0.8),))
        self.X, _ = train_set[self.random_index]  
        self.volume = hypersphere_volume(dimensions=9, radius=radius)
        self.include_volume = True

    def compute(self, classifier_out):
        """
        Perturbation must be of dimension 100, 500, 5 namely P, S, F
        Where P is the number of perturbation
        S is the number of sample
        F is the number of features
        Sample must be of dimensio S, F
        """ 
        sample_perturbed = self.X.to("cuda") + self.perturbation.unsqueeze(1).repeat(1, self.X.shape[0], 1)
        out = self.function(sample_perturbed)
        out = torch.argmax(out, dim=-1)
        classifier_out = classifier_out[self.random_index]
        classifier_out = classifier_out.unsqueeze(1)
        classifier_out = classifier_out.repeat(1, self.n_samples)
        classifier_out = classifier_out.to("cuda")
        classifier_out = classifier_out.permute(1, 0)
        different = out != classifier_out
        average_function_value = torch.sum(different, dim=0)/different.shape[0]
        value = self.volume * average_function_value if self.include_volume else average_function_value
        
        
        return torch.mean(value).cpu(), torch.std(value).cpu()
    
    def counterfactual(self, X, target):
        """
        Perturbation must be of dimension 100, 500, 5 namely P, S, F
        Where P is the number of perturbation
        S is the number of sample
        F is the number of features
        Sample must be of dimensio S, F
        """ 
        sample_perturbed = X.to("cuda") + self.perturbation.unsqueeze(1).repeat(1, X.shape[0], 1)
        out = self.function(sample_perturbed)
        #out = torch.argmax(out, dim=-1)
        out = out.reshape((out.shape[0] * out.shape[1], 2))
        #classifier_out = classifier_out[self.random_index]
        target = torch.argmax(target, dim=-1)
        target = target.unsqueeze(1)
        target = target.repeat(1, self.n_samples)
        target = target.reshape(out.shape[0])
        target = target.to("cuda")
        
        
        return out, target
      