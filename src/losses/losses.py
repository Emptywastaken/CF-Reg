from torch.nn import Module
import torch

class CounterfactualRegularizationLoss(Module):
    
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.counterfactual_loss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
    
    def forward(self, input, target, out_cf, target_cf):
        """
        out dimension: N, C
        target dimension: N
        out_cf dimension: N, S, C
        target_cf: N, S, C
        
        """
        train_loss = self.train_loss(input, target)
        counterfactual_loss = self.counterfactual_loss(out_cf, target_cf)
        
        return train_loss + self.alpha * counterfactual_loss
    
class DynamicCounterfactualRegularizationLoss(Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.counterfactual_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, input: torch.tensor, target: torch.tensor, out_cf: torch.tensor, target_cf: torch.tensor):
        """
        out dimension: N, C
        target dimension: N
        out_cf dimension: N, S, C
        target_cf: N, S, C
        
        """
        train_loss = self.train_loss(input, target)
        counterfactual_loss = self.counterfactual_loss(out_cf, target_cf)
        predicted_class_cf = torch.argmax(out_cf, dim=1)
        alpha = (target_cf != predicted_class_cf).sum() / torch.numel(predicted_class_cf)
        return train_loss + alpha * counterfactual_loss