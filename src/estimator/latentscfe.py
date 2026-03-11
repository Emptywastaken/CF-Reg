import torch
from torch import Tensor
from .estimator import Estimator

class LatentSCFEEstimator(Estimator):
    """
    Estimator that calculates the margin distance to the linear decision boundary
    in the latent space (i.e. the last hidden layer before the linear classification head).
    """
    def __init__(self, function: torch.nn.Module, **kwargs):
        self.function = function
        self.reg_coef = kwargs.get('reg_coef')
        self.w_norm_history = []

    def get_estimate(self, data: Tensor, output: Tensor) -> Tensor:
        """
        Finds the distance to the closest counterfactual, which in this latent space is:
        abs((w^T z + b) / ||w||_2)
        
        Args:
            data (Tensor): Original input batch (not used explicitly here since output avoids re-forwarding).
            output (Tensor): Raw logits / linear output from the model. 
                             Corresponds exactly to w^T z + b.

        Returns:
            Tensor: A batch of latent space distances.
        """
        # Get the weight matrix for the model's final classification layer
        w = self.function.get_last_layer_weight()
        
        # We calculate the L2 norm (Euclidean length) of the entire weight matrix.
        # By default, torch.norm with p=2 will compute the Frobenius norm over all dimensions 
        # of the tensor efficiently at the C++ level without needing us to explicitly specify dimensions.
        w_norm = torch.norm(w, p=2)  
        self.w_norm_history.append(w_norm.item())
        
        # Finally, the distance to the counterfactual is just the absolute output divided by the weight's norm
        # Reverting to the geometric normalized distance but adding a stabilization constant (epsilon).
        # This explicitly stops the model from shrinking w_norm infinitesimally close to 0 to blow up the distance.
        epsilon = 1e-3
        distance = torch.abs(output) / (w_norm + epsilon)
        
        return distance

    def get_estimate_name(self) -> str:
        return "latent_scfe"

    def build_log(self, values: list, stage: str) -> dict:
        import numpy as np

        if not values:
            return {}
            
        max_value = max(values)
        mean_value = np.mean(values)
        first_quartile = np.percentile(values, 25)
        third_quartile = np.percentile(values, 75)
        median_value = np.median(values)
        min_value = min(values)

        log_data = {
            f"{stage}/max latent_distance": max_value,
            f"{stage}/mean latent_distance": mean_value,
            f"{stage}/first_quartile latent_distance": first_quartile,
            f"{stage}/third_quartile latent_distance": third_quartile,
            f"{stage}/median latent_distance": median_value,
            f"{stage}/min latent_distance": min_value,
        }
        
        if hasattr(self, 'w_norm_history') and self.w_norm_history:
            log_data[f"{stage}/w_norm"] = np.mean(self.w_norm_history)
            self.w_norm_history.clear()

        return log_data
