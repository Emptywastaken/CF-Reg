from typing import List
import torch
import numpy as np
from torch import Tensor
from torch.linalg import solve
from .estimator import Estimator
from torch.utils.data import TensorDataset
import dice_ml
from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
import pandas as pd


class DiceEstimator(Estimator):
    def __init__(self,
                 function: torch.nn.Module,
                 train_set: TensorDataset,
                 continuous_features: list[int] = None,
                 outcome_name: str = "target",
                 backend: str = "PYT",
                 cf_per_instance: int = 1,
                 desired_class: str = "opposite",
                 **kwargs):
        """
        function: your torch model (in eval mode).
        train_dataset: TensorDataset(features, labels)
        continuous_features: list of column‐indices (0…D−1) to treat as continuous.
        outcome_name: name of the label column we’ll create.
        backend: "PYT" for PyTorch.
        cf_per_instance: number of CFs to generate per sample.
        desired_class: "opposite" or a specific class label.
        gradient_args: extra kwargs for DiCE's gradient method.
        """
        super().__init__()
        self.function        = function.eval()
        self.cf_per_instance = cf_per_instance
        self.desired_class   = desired_class
        self.dice_min_iter   = kwargs.pop("dice_min_iter", 50)
        self.dice_max_iter   = kwargs.pop("dice_max_iter", 500)
        self.dice_yloss_type = kwargs.pop("yloss_type", "hinge_loss")
        self.dice_learning_rate   = kwargs.pop("learning_rate", 0.05)
        self.dice_limit_steps_ls   = kwargs.pop("dice_limit_steps_ls", 1000)


        # 1) Unpack your TensorDataset into numpy arrays
        feats, labels = train_set.tensors
        feats_np = feats.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if continuous_features == None:
            continuous_features = list(range(0, feats_np.shape[1]-1))

        # 2) Build feature names and DataFrame for DiCE
        D = feats_np.shape[1]
        feature_names = [f"f{i}" for i in range(D)]
        df = pd.DataFrame(feats_np, columns=feature_names)
        df[outcome_name] = labels_np

        # 3) Data interface
        self.data_interface = dice_ml.Data(
            dataframe=df,
            continuous_features=[feature_names[i] for i in continuous_features],
            outcome_name=outcome_name,
        )

        # 4) Model interface
        self.model_interface = dice_ml.Model(
            model=self.function,
            backend=backend,
            model_type="classifier",
        )
        
        # 5) The DiCE explainer
        self.dice = DicePyTorch(
            data_interface=self.data_interface,
            model_interface=self.model_interface
        )

        # store feature names for query‐time
        self.feature_names = feature_names

    @torch.no_grad()
    def _generate_counterfactuals(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] tensor of new queries
        returns: [N, D] counterfactuals
        """
        # 1) to DataFrame
        df_query = pd.DataFrame(
            x.detach().cpu().numpy(),
            columns=self.feature_names
        )

        # 2) generate CFs
        cf_obj = self.dice.generate_counterfactuals(
            df_query,
            total_CFs=self.cf_per_instance,
            desired_class=self.desired_class,
            yloss_type = self.dice_yloss_type,
            min_iter = self.dice_min_iter,
            max_iter = self.dice_max_iter,
            learning_rate = self.dice_dice_learning_rate,
            limit_steps_ls = self.dice_limit_steps_ls
        )

        # 3) extract first CF per instance
        cfs_df = cf_obj.cf_examples_list[0].final_cfs_df
        # shape is [N * cf_per_instance, D], in same order as df_query
        cfs_np = cfs_df[self.feature_names].values \
                 .reshape(len(df_query), self.cf_per_instance, -1)[:, 0, :]

        return torch.from_numpy(cfs_np).to(x.device).type_as(x)

    def get_estimate_name(self) -> str:
        return "DICE"

    def get_estimate(self, data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Returns per‑sample distances ‖x - x_ce‖₂, shape [N].
        """
        x_ce = self._generate_counterfactuals(data)

        # Compute L2 distance over all non‑batch dims
        with torch.enable_grad():
            dist = torch.norm(data - x_ce, p=2, dim=tuple(range(1, data.ndim)))
            mask = torch.isnan(dist)
            if mask.any():
                # e.g. use the largest finite distance in the batch
                fallback = dist[~mask].max().detach()
                dist[mask] = fallback
        return dist

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