import torch
from torch import Tensor
from .estimator import Estimator
''' 
DISCLAIMER: LATENT DISTANCE SHORTCUT FOR MULTICLASS
 This multiclass distance calculation relies on the logit difference: |logit_c - logit_k|.
 This mathematical shortcut is ONLY valid if the final layer is strictly linear (affine).
 DO NOT use this implementation if Dropout, BatchNorm, or ReLUs are applied 
 directly between the latent vector 'z' and the final linear classifier.
'''
class LatentSCFEEstimator(Estimator):
    """
    Estimator that calculates the margin distance to the linear decision boundary
    in the latent space (i.e. the last hidden layer before the linear classification head).
    """
    def __init__(self, function: torch.nn.Module, **kwargs):
        self.function = function
        self.epsilon = kwargs.get('epsilon', 0.0)
      
    def get_estimate(self, data: Tensor, output: Tensor, target: Tensor = None) -> Tensor:
        """
        Finds the distance to the closest counterfactual, which in this latent space is:
        abs((w^T z + b) / ||w||_2) for binary
        or ((w_c - w_k)^T z + (b_c - b_k)) / ||w_c - w_k||_2 for multiclass.
        
        Args:
            data (Tensor): Original input batch (not used explicitly here since output avoids re-forwarding).
            output (Tensor): Raw logits / linear output from the model. 

        Returns:
            Tensor: A batch of latent space distances.
        """
        # Get the weight matrix for the model's final classification layer
        w = self.function.get_last_layer_weight()
        
        if w.shape[0] == 1:
            # Binary classification
            w_norm = torch.norm(w, p=2)  
            distance = torch.abs(output) / (w_norm + self.epsilon)
        else:
            # Multiclass classification
            logits = output
            targets = target
            weights = w
            
            batch_size, num_classes = logits.shape

            # 1. Gather the logits for the true classes
            # shape: [batch_size, 1]
            logit_true = logits.gather(1, targets.view(-1, 1))

            # 2. Compute the numerator: |logit_true - logit_k| for all classes
            # Broadcasting [batch_size, 1] - [batch_size, num_classes]
            logit_diff = torch.abs(logit_true - logits)

            # 3. Gather the weights for the true classes
            # shape: [batch_size, latent_dim]
            w_true = weights[targets]

            # 4. Compute the denominator: ||w_true - w_k|| for all classes
            # Broadcasting [batch_size, 1, latent_dim] - [1, num_classes, latent_dim]
            w_diff = w_true.unsqueeze(1) - weights.unsqueeze(0)
            # Use torch.sqrt(sum(sq) + 1e-8) to avoid NaN gradients at exactly 0 when w_diff is the true class difference
            w_diff_norm = torch.sqrt(torch.sum(w_diff ** 2, dim=2) + 1e-8) # shape: [batch_size, num_classes]

            # 5. Calculate full geometric distance
            distances = logit_diff / (w_diff_norm + self.epsilon)

            # 6. Mask out the true class (distance to itself is 0/0)
            mask = torch.ones_like(distances, dtype=torch.bool)
            mask[torch.arange(batch_size), targets] = False
            distances.masked_fill_(~mask, float('inf'))

            # 7. Find the closest counterfactual distance for each sample in the batch
            distance, _ = torch.min(distances, dim=1)

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
        
        
        with torch.no_grad():
            w = self.function.get_last_layer_weight()
            current_w_norm = torch.norm(w, p=2).item()
            log_data[f"{stage}/w_norm"] = current_w_norm

        return log_data
