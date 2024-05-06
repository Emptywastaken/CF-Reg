import numpy as np
import matplotlib.pyplot as plt
import math 
import torch

def hypersphere_volume(dimensions: int, radius: float)->float:

    pi_pow = np.pi**(dimensions/2)
    euler_argument = (dimensions/2) + 1
    euler_value = math.gamma(euler_argument)
    result = (pi_pow/euler_value)*(radius**dimensions)

    return result


def random_points_in_n_sphere(num_points: int, n: int, radius: float)->np.array:

    points = np.random.randn(num_points, n)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    scales = np.random.rand(num_points) ** (1/n)
    points = points * scales[:, np.newaxis]
    points *= radius
    return torch.tensor(points, device="cuda", dtype=torch.float32)

