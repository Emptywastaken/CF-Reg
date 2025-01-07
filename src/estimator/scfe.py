import torch
import numpy as np
from torch import Tensor
from torch.linalg import solve
from .estimator import Estimator


class SCFEEstimator(Estimator):
    def __init__(self, 
                 function: torch.nn.Module,
                 reg_coef: float):
        self.function = function
        self.reg_coef = reg_coef

    def get_estimate(self, data: Tensor, s: Tensor) -> Tensor:
        if len(s.shape) == 1:
            # Simplified binary classification case
            return self._get_estimate_binary(data, s)  # Pass single target score for both classes
        else:
            # General multi-class case
            return self._get_estimate_multiclass(data, s)

    def _get_estimate_binary(self, data: Tensor, s: Tensor) -> Tensor:
        import time
        """
        Computes the optimal counterfactual perturbation for a batch of inputs, 
        simplifying for the case where nclasses = 2 and target scores are symmetric.

        Args:
            data (torch.Tensor): Input batch of shape [batch_size, input_dim].
            s (torch.Tensor): Target scores for the batch, shape [batch_size].

        Returns:
            torch.Tensor: Optimal perturbations for the batch, shape [batch_size, input_dim].
        """
        # Step 1: Use the `linearize` method to compute the Jacobian and output
        lin_result = self.function.linearize(data)
        #jacobian = lin_result["gradient"]  # [batch_size, nclasses, input_dim]         #TODO
        w = lin_result["gradient"]  # [batch_size, nclasses, input_dim] 
        output = lin_result["output"]  # [batch_size, nclasses]

        # Step 2: Convert in a single output function with the same decision boundary
        #output = output[:, 1] - output[:, 0]    # Definition of a new function h(x) = f_1(x) - f_0(x)       #TODO
        #w = jacobian[:, 1] - jacobian[:, 0]  # Difference of class gradients, shape [batch_size, input_dim] #TODO

        # Step 3: Compute the residual m = s - f(x)
        m = s - output  # Use the first class residual for simplification, shape [batch_size]

        # Step 4: Apply the Sherman-Morrison formula for δ*_SCFE
        w = w.expand(m.shape[0], -1)
        w_norm_squared = torch.norm(w, p=2, dim=1) ** 2 # [batch_size], ∥w∥²

        #print("w.shape: ", w.shape)
        #print("s.shape: ", s.shape)
        #print("output.shape: ", output.shape)
        #print("m.shape: ", m.shape)
        #print("w_norm_squared.shape: ", w_norm_squared.shape)
        #print("w.shape", w.shape)
        #print("(m * self.reg_coef / (self.reg_coef + w_norm_squared)).shape", (m * self.reg_coef / (self.reg_coef + w_norm_squared)).shape)
        #print("(m * self.reg_coef / (self.reg_coef + w_norm_squared)).unsqueeze(1).shape", (m * self.reg_coef / (self.reg_coef + w_norm_squared)).unsqueeze(1).shape)

        delta_scfe = (m * self.reg_coef / (self.reg_coef + w_norm_squared)).unsqueeze(1) * w  # [batch_size, input_dim]
        #print("delta_scfe.shape: ", delta_scfe.shape)
        norm_delta_scfe = torch.norm(delta_scfe, p=2, dim=1)

        return norm_delta_scfe

    def _get_estimate_multiclass(self, data: Tensor, s: Tensor) -> Tensor:
        """
        Computes the optimal counterfactual perturbation for a batch of inputs, 
        considering multi-class outputs.

        Args:
            data (torch.Tensor): Input batch of shape [batch_size, input_dim].
            s (torch.Tensor): Target scores for the batch, shape [batch_size, nclasses].

        Returns:
            torch.Tensor: Optimal perturbations for the batch, shape [batch_size, input_dim].
        """
        # Step 1: Use the `linearize` method to compute the Jacobian and output
        lin_result = self.function.linearize(data)
        jacobian = lin_result["gradient"]  # [batch_size, nclasses, input_dim]
        output = lin_result["output"]  # [batch_size, nclasses]

        # Step 2: Compute the residual m = s - f(x) for each class
        m = s - output  # [batch_size, nclasses]

        # Step 3: Compute the optimal perturbation δ*_SCFE for the batch
        batch_size, nclasses, input_dim = jacobian.shape

        # Compute ∥w∥² for each class
        w_norm_squared = torch.einsum('bci,bci->bc', jacobian, jacobian)  # [batch_size, nclasses]

        # Compute δ*_SCFE for each class using Sherman-Morrison formula
        delta_per_class = (m / (self.reg_coef + w_norm_squared)).unsqueeze(-1) * jacobian  # [batch_size, nclasses, input_dim]

        # Aggregate contributions across classes
        delta_scfe = delta_per_class.sum(dim=1)  # [batch_size, input_dim]

        return delta_scfe
        

    def get_estimate_name(self) -> str:
        """
        Implementation of get_estimate_name abstract method.
        """
        return "scfe"
        
    def build_log(self, values, stage):
        import numpy as np

        # Calculate the required statistics
        max_value = max(values)
        mean_value = np.mean(values)
        first_quartile = np.percentile(values, 25)
        third_quartile = np.percentile(values, 75)
        median_value = np.median(values)
        min_value = min(values)

        # Construct the dictionary with keys based on `stage` and metrics
        log_data = {
            f"{stage}/max delta$": max_value,
            f"{stage}/mean delta$": mean_value,
            f"{stage}/first_quartile delta$": first_quartile,
            f"{stage}/third_quartile delta$": third_quartile,
            f"{stage}/median delta$": median_value,
            f"{stage}/min delta$": min_value,
        }

        return log_data

