from torch.nn import Module
import torch
from  src.losses.losses import CounterfactualRegularizationLoss

def get_loss(name: str) -> Module:
    
    if name == "regularized":
        
        return CounterfactualRegularizationLoss
    
    elif name == "normal":
        
        return  torch.nn.CrossEntropyLoss
    
    else:
        
        raise ValueError(f"This loss has not been implemented yet!")