from torch.nn import Module
import torch
from  src.losses.losses import CounterfactualRegularizationLoss, DynamicCounterfactualRegularizationLoss, SCFERegularizationLoss, L1CrossEntropy, L2CrossEntropy, CrossEntropy 
def get_loss(**kwargs) -> Module:
    
    name: str = kwargs.pop("type")
    
    if name == "regularized":
        """
        Takes as input alpha, a float number [0, +inf)
        """
        
        return CounterfactualRegularizationLoss(**kwargs)
    
    elif name == "dyn_regularized":
        
        return DynamicCounterfactualRegularizationLoss()
    
    elif name == "normal":
        return CrossEntropy(**kwargs)
    
    elif name == "scfe_regularization":
        return SCFERegularizationLoss(**kwargs)

    elif name == "l1normal":
        return L1CrossEntropy(**kwargs)
    
    elif name == "l2normal":
        return L2CrossEntropy(**kwargs)

    else:
        
        raise ValueError(f"This loss has not been implemented yet!")