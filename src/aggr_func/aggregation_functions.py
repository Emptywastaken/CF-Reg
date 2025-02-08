import torch
from torch.nn import Module
from ..estimator import MontecarloEstimator

def get_aggr_func(**kwargs):
    aggr_func = kwargs.pop('aggr_func') #dict
    type = aggr_func['type']
    if type == "mean":
        return Mean()
    
    elif type == "montecarlo_vcp_weighted_mean":
        vcp_weighted_mean = Montecarlo_vcp_weighted_mean(**(aggr_func | kwargs))
        return vcp_weighted_mean
    
    return None

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

