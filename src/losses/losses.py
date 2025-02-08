from torch.nn import Module
import torch
from ..aggr_func.aggregation_functions import get_aggr_func

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
    def __init__(self, alpha: float = 0.1, binary: bool = True, **kwargs) -> None:
        super().__init__()
        self.binary = binary
        if self.binary: 
            self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.train_loss = torch.nn.functional.cross_entropy
        self.alpha = alpha
        self.aggr_function = get_aggr_func(**kwargs)


    def forward(self, **kwargs):
        """ input : model's predictions
            target: true classes
        """
        #print(kwargs)
        input : torch.Tensor = kwargs['input']
        target : torch.Tensor = kwargs['target']

        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1,"Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"
        #assert estimate.dim() == 1, "Estimate must be 1D"
        #print("input.dtype: ", input.dtype)
        #print("target.dtype: ", target.dtype)
        train_loss = self.train_loss(input, target)
        reg_term = self.aggr_function(**kwargs)
        return train_loss + self.alpha * reg_term
    

class L1CrossEntropy(Module):
    def __init__(self, alpha: float = 0.1, binary: bool = True) -> None:
        super().__init__()
        self.binary = binary
        if self.binary: 
            self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.train_loss = torch.nn.functional.cross_entropy
        self.alpha = alpha
    

    def forward(self, **kwargs):
        """ input : model's predictions
            target: true classes
        """
        input : torch.Tensor = kwargs['input']
        target : torch.Tensor = kwargs['target']
        weights: torch.Tensor = kwargs['weights']
        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1,"Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"

        train_loss = self.train_loss(input, target)
        l1_reg = sum(param.abs().sum() for param in weights)
        train_loss += self.alpha * l1_reg
        return train_loss
    
class L2CrossEntropy(Module):
    def __init__(self, alpha: float = 0.1, binary: bool = True) -> None:
        super().__init__()
        self.binary = binary
        if self.binary: 
            self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.train_loss = torch.nn.functional.cross_entropy
        self.alpha = alpha
    


    def forward(self, **kwargs):
        """ input : model's predictions
            target: true classes
        """
        input : torch.Tensor = kwargs['input']
        target : torch.Tensor = kwargs['target']
        weights: torch.Tensor = kwargs['weights']
        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1,"Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"

        train_loss = self.train_loss(input, target)
        l2_reg = sum(param.pow(2).sum() for param in weights)
        return train_loss + self.alpha * l2_reg
    
class CrossEntropy(Module):
    def __init__(self, binary: bool = True) -> None:
        super().__init__()
        self.binary = binary
        if self.binary: 
            self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.train_loss = torch.nn.functional.cross_entropy

    def forward(self, **kwargs):
        """ input : model's predictions
            target: true classes
        """
        input : torch.Tensor = kwargs['input']
        target : torch.Tensor = kwargs['target']
        return self.train_loss(input, target)
    