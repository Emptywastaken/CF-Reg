
from src.functions.abstract_function import GenericFunction
from src.utility.geometric import random_points_in_n_sphere, hypersphere_volume
import torch
import numpy as np
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

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
        pca = False
        p_x_list = []

        for i in range(len(self.random_index.tolist())):

            sample_perturbed = self.X[i].to("cuda") + self.perturbation
            out = self.function(sample_perturbed)
            out = torch.argmax(out, dim=1)
            mask = (out != classifier_out[i].repeat(self.n_samples).to("cuda")).cpu()

            if torch.sum(mask) != self.n_samples and torch.sum(mask) != 0:
                pca = False
            if pca:
                x_pca = self.pca(torch.cat([sample_perturbed, self.X[i].unsqueeze(0).to("cuda")]), n_components=3).cpu()
                self.plot_pca(x_pca, mask)
                pca=False

            average_function_value = torch.sum(out != classifier_out[i].repeat(self.n_samples).to("cuda"))/torch.numel(out)
            value = self.volume * average_function_value if self.include_volume else average_function_value
            p_x_list.append(value)
        
        return torch.mean(torch.tensor(p_x_list)), torch.std(torch.tensor(p_x_list))
    

    def pca(self, X, n_components: int =2):
        
        # 1. Standardize the data
        X_mean = torch.mean(X, 0)
        X_std = torch.std(X, 0)
        X = (X - X_mean) / X_std

        # 2. Calculate the covariance matrix
        X_t = X.t()
        covariance_matrix = X_t @ X / (X.size(0) - 1)

        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # 4. Sort eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Select the top n_components eigenvectors (n_components <= number of features)
        components = eigenvectors[:, :n_components]

        # 6. Transform the original matrix
        X_pca = X @ components

        return X_pca
    
    def plot_pca(x_pca, mask):

        x_pca = x_pca[:-1, :]
        original_sample = x_pca[-1, :]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        ax.scatter(x_pca[mask, 0], x_pca[mask, 1], x_pca[mask, 2], c="blue", label="Counterfactual", marker="*")
        ax.scatter(x_pca[~mask, 0], x_pca[~mask, 1], x_pca[~mask, 2], c="red", label="Factual")
        ax.scatter(original_sample[0], original_sample[1], original_sample[2], c="green", s=50, label="Original Sample")
        # Setting labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'Random Points Inside a 3D Sphere')
        plt.legend()
        # Show the plot
        plt.show()