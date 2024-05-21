import numpy as np
import matplotlib.pyplot as plt
import math 
import torch

from typing import Tuple


class Sphere:
    
    
    @staticmethod
    def hypersphere_volume(dimensions: int, radius: float = 1.0) -> float:
        from scipy.special import gamma
        
        pi_pow = np.pi**(dimensions/2)
        euler_argument = (dimensions/2) + 1
        euler_value = gamma(euler_argument)
        result = (pi_pow/euler_value)*(radius**dimensions)

        return result

    @staticmethod
    def random_points_in_sphere(num_points: int, 
                                  shape: Tuple[int, ...], 
                                  radius: float = 1.0, 
                                  device: str = "cuda") -> torch.Tensor:
        """
        Generate random points inside an n-dimensional sphere of given radius with uniform distribution.

        Parameters:
        - num_points (int): Number of points to generate.
        - shape (Tuple[int, ...]): Shape of each point (e.g., (28, 28) for a 28x28 matrix).
        - radius (float): Radius of the sphere.
        - device (str): Device used (cuda or cpu)

        Returns:
        - torch.Tensor: A tensor of shape (num_points, *shape) containing the generated points.
        """
        total_dim = np.prod(shape)
        points = 2 * np.random.rand(num_points, total_dim) - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        scales = 2 * np.random.rand(num_points, 1) - 1
        points = points * scales
        points *= radius
        points = points.reshape(num_points, *shape)
        
        return torch.tensor(points, device=device, dtype=torch.float32)



if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    sphere = Sphere()
    
    points = sphere.random_points_in_n_sphere(num_points=10000, shape=(28, 28), radius=1)

    points = points.cpu()
    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # Setting labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'Random Points Inside a 3D Sphere with Radius {1}')

    # Show the plot
    plt.show()