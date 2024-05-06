
from src.functions.abstract_function import GenericFunction
from src.utility.geometric import random_points_in_n_sphere, hypersphere_volume
import torch
import numpy as np
from torch.utils.data import TensorDataset

class MontecarloEstimator():

    def __init__(self, function, train_set: TensorDataset, n_samples: int = 1000, **kwargs) -> None:
        self.function = function
        self.n_samples = n_samples
        self.perturbation = random_points_in_n_sphere(num_points=n_samples, n=9, radius=1)
        self.random_index = torch.randint(len(train_set), (int(len(train_set)*0.8),))
        
        self.X, self.y = train_set[self.random_index]  
        self.volume = hypersphere_volume(dimensions=9, radius=1)

        pass
        

    def compute(self, sample, target):

        p_x_list = []

        for i in range(self.X.shape[0]):

            sample_perturbed = self.X[i].to("cuda") + self.perturbation

            out = self.function(sample_perturbed)

            out = torch.argmax(out, dim=1)

            average_function_value = torch.sum(out != self.y[i].repeat(self.n_samples).to("cuda"))/torch.numel(out)


            p_x_list.append(self.volume*average_function_value)
        
        return torch.mean(torch.tensor(p_x_list)), torch.std(torch.tensor(p_x_list))