from torch.nn import Module
import torch
from  src.losses.losses import CounterfactualRegularizationLoss, DynamicCounterfactualRegularizationLoss, SCFERegularizationLoss
def get_loss(**kwargs) -> Module:
    
    name: str = kwargs.pop("type")
    
    if name == "regularized":
        """
        Takes as input alpha, a float number [0, +inf)
        """
        
        return CounterfactualRegularizationLoss(**kwargs)
    
    if name == "dyn_regularized":
        
        return DynamicCounterfactualRegularizationLoss()
    
    elif name == "normal":
        binary: bool = kwargs.pop("binary")
        if binary:
            return torch.nn.functional.binary_cross_entropy_with_logits
        else:
            return  torch.nn.functional.cross_entropy
    
    elif name == "scfe_regularization":
        return SCFERegularizationLoss(**kwargs)

    else:
        
        raise ValueError(f"This loss has not been implemented yet!")