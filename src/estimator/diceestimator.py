from .estimator import Estimator
import torch
from torch.utils.data import TensorDataset
from src.utility.dice import dice_cf_set_batch

class DiceEstimator(Estimator):
    def __init__(self, 
        function: torch.nn.Module, 
        K=1, lambda_proximity=0.1, 
        gamma_diversity=0.1,
        num_steps=500, learning_rate=0.01,
        loss_type='hinge', prox_type='mad'
    ):
        """
        DiceEstimator.

        Args:
            function: PyTorch model returning a logit per sample.
            K: number of counterfactual sets.
            lambda_proximity: weight for proximity loss.
            gamma_diversity: weight for diversity.
            num_steps: number of optimization steps.
            lr: learning rate.
            loss_type: 'hinge' or 'bce'.
            prox_type: 'l1' or 'mad'.
        """
        super().__init__()
        self.function = function
        self.K = K
        self.lambda_proximity = lambda_proximity
        self.gamma_diversity = gamma_diversity
        self.num_steps = num_steps
        self.lr = learning_rate
        self.loss_type = loss_type
        self.prox_type = prox_type
        
    def get_estimate_name(self) -> str:
        return "dice"
    
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
    
    def get_estimate(self, data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Get the estimate of the difference between each sample in data and its counterfactual example.

        Args:
            data (torch.Tensor): Input batch of shape [batch_size, input_dim].
            output (torch.Tensor): Output predictions from the model, shape [batch_size, nclasses].

        Returns:
            torch.Tensor: Tensor of norms of the difference between each sample in data and its counterfactual example.
        """
        cfs = dice_cf_set_batch(
            model = self.function, 
            x = data, 
            logits = output, 
            K=self.K, 
            lambda_proximity=self.lambda_proximity, 
            gamma_diversity=self.gamma_diversity,
            num_steps=self.num_steps, 
            lr=self.lr, 
            loss_type=self.loss_type, 
            prox_type=self.prox_type
        )

        # Assume:
        # - cfs is a list of tensors, each of shape (K, input_dim) for each input (len(cfs) == batch_size)
        # - data is a tensor of shape (batch_size, input_dim)
        # 1) stack your K tensors into one tensor of shape [K, B, D]
        cfs_stack = torch.stack(cfs, dim=0)            # -> [4, 128, 10]

        # 2) broadcast-subtract data from each of the K slices
        #    data.unsqueeze(0) has shape [1, 128, 10], which broadcasts to [4,128,10]
        diff = cfs_stack - data.unsqueeze(0)           # -> [4, 128, 10]

        # 3) compute the L2 norm across the D-dimension
        #    .norm(dim=2) → [4, 128]
        dists = diff.norm(p = 2, dim=2)                       # -> [4, 128]

        # 4) take the mean over the K-dimension to get one distance per batch
        mean_dists = dists.mean(dim=0)  
        assert mean_dists.shape == (data.shape[0],), "Output shape mismatch"
        return mean_dists