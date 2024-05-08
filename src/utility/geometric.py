import numpy as np
import matplotlib.pyplot as plt
import math 
import torch
from scipy.special import gamma

def hypersphere_volume(dimensions: int, radius: float) -> float:

    pi_pow = np.pi**(dimensions/2)
    euler_argument = (dimensions/2) + 1
    euler_value = gamma(euler_argument)
    result = (pi_pow/euler_value)*(radius**dimensions)

    return result


def random_points_in_n_sphere(num_points: int, n: int, radius: float) -> np.array:

    points = np.random.randn(num_points, n)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    scales = np.random.rand(num_points) ** (1/n)
    points = points * scales[:, np.newaxis]
    points *= radius

    return torch.tensor(points, device="cuda", dtype=torch.float32)



if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D

    points = random_points_in_n_sphere(num_points=10000, n=3, radius=1)

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