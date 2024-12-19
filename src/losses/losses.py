from torch.nn import Module
import torch
from ..estimator import SCFEEstimator 

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
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, out_cf: torch.Tensor, target_cf: torch.Tensor):
        """
        out dimension: N, C
        target dimension: N
        out_cf dimension: N, S, C
        target_cf: N, S, C
        
        """
        train_loss = self.train_loss(input, target)
        counterfactual_loss = self.counterfactual_loss(out_cf, target_cf)
        predicted_class_cf = torch.argmax(out_cf, dim=1)
        alpha = (target_cf != predicted_class_cf).sum() / torch.numel(predicted_class_cf)   #TODO be aware that this element cannot contribute to a loss function since is detached from the computational graph
        return  train_loss + alpha * counterfactual_loss
    
class SCFERegularizationLoss(Module):
    def __init__(self, alpha: float = 0.1, binary: bool = True) -> None:
        super().__init__()
        self.binary = binary
        if self.binary: 
            self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.train_loss = torch.nn.functional.cross_entropy
        self.alpha = alpha
    

    def forward(self, input: torch.Tensor, target: torch.Tensor, estimate: torch.Tensor):
        """ input : model's predictions
            target: true classes
        """
        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1,"Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"
        assert estimate.dim() == 1, "Estimate must be 1D"
        train_loss = self.train_loss(input, target)
        return train_loss + self.alpha * torch.mean(estimate) 