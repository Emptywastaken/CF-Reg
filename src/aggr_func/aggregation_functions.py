import torch
from torch.nn import Module
from ..estimator import MontecarloEstimator

def get_aggr_func(**kwargs):
    aggr_func = kwargs.pop('aggr_func') #dict
    type = aggr_func['type']
    if type == "mean":
        print("Aggregation Function: mean")
        return Mean()
    
    elif type == "montecarlo_vcp_weighted_mean":
        vcp_weighted_mean = Montecarlo_vcp_weighted_mean(**(aggr_func | kwargs))
        return vcp_weighted_mean
    
    elif type == "tp_tn_mean":
        print("Aggregation Function: TP_TN_mean")
        return TP_TN_mean()
    
    elif type == "fp_fn_mean":
        print("Aggregation Function: FP_FN_mean")
        return FP_FN_mean()
    
    elif type == "supervised_mean":
        print("Aggregation Function: Supervised_mean")
        return Supervised_mean()
    else :
        raise ValueError(f"Aggregation function {type} is not available!")

class Mean(Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        estimate:torch.Tensor = kwargs.pop("estimate")
        return torch.mean(estimate)

class Montecarlo_vcp_weighted_mean(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.montecarlo = MontecarloEstimator(**kwargs)

    def forward(self, **kwargs):
        data: torch.Tensor = kwargs.pop("data")
        output: torch.Tensor = kwargs.pop("input") # model output are "input" in loss signature
        estimate:torch.Tensor = kwargs.pop("estimate")

        torch.set_grad_enabled(False)
        vcp = self.montecarlo.get_estimate(data=data, output=output)
        torch.set_grad_enabled(True)

        assert estimate.shape == vcp.shape, "estimate and vcp must have the same shape"

        sum_vcp = torch.sum(vcp)

        # Ensure numerical stability using torch.where instead of returning a constant 0
        weighted_sum = torch.sum(estimate * vcp) / torch.where(sum_vcp != 0, sum_vcp, torch.tensor(1.0, device=sum_vcp.device))
        return weighted_sum

class TP_TN_mean(Module):
    def __init__(self, **kwargs):
        super().__init__()

    
    def forward(self, **kwargs):
        output: torch.Tensor = kwargs.pop("input")  # Model output, named "input" in loss signature
        target: torch.Tensor = kwargs.pop("target")
        estimate: torch.Tensor = kwargs.pop("estimate")
        
        torch.set_grad_enabled(False)
        
        # Compute mask efficiently
        mask = ((output < 0) & (target == 0)) | ((output >= 0) & (target == 1))
        mask = mask.float()  # Convert to float tensor for weighting
        torch.set_grad_enabled(True)
        # Compute the weighted mean of estimate
        weighted_mean = (estimate * mask).sum() / mask.sum().clamp(min=1)  # Avoid division by zero
        
        
        
        return weighted_mean
            
class FP_FN_mean(Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        output: torch.Tensor = kwargs.pop("input")  # Model output, named "input" in loss signature
        target: torch.Tensor = kwargs.pop("target")
        estimate: torch.Tensor = kwargs.pop("estimate")
        
        torch.set_grad_enabled(False)
        
        # Compute mask with 0 or -1
        mask = -((output < 0) & (target == 1) | (output >= 0) & (target == 0)).float()
        torch.set_grad_enabled(True)
        # Compute the weighted mean of estimate
        weighted_mean = (estimate * mask).sum() / mask.abs().sum().clamp(min=1)  # Avoid division by zero
        
        
        
        return weighted_mean
    
class Supervised_mean(Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        output: torch.Tensor = kwargs.pop("input")  # Model output, named "input" in loss signature
        target: torch.Tensor = kwargs.pop("target")
        estimate: torch.Tensor = kwargs.pop("estimate")
        
        torch.set_grad_enabled(False)
        
        # Compute mask with 1 or -1
        mask = 2 * (((output < 0) & (target == 0)) | ((output >= 0) & (target == 1))).float() - 1
        torch.set_grad_enabled(True)
        # Compute the weighted mean of estimate
        weighted_mean = (estimate * mask).sum() / mask.abs().sum().clamp(min=1)  # Avoid division by zero
        
        
        
        return weighted_mean