from torch.nn import Module
import torch

class CounterfactualRegularizationLoss(Module):
    
    def __init__(self) -> None:
        
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.counterfactual_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, out, target, out_cf, target_cf):
        """
        out dimension: N, C
        target dimension: N
        out_cf dimension: N, S, C
        target_cf: N, S, C
        
        """
        train_loss = self.train_loss(out, target)
        counterfactual_loss = self.counterfactual_loss(out_cf, target_cf)
        
        return train_loss + counterfactual_loss