from torch.nn import Module
import torch
from ..aggr_func.aggregation_functions import get_aggr_func

class CounterfactualRegularizationLoss(Module):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        alpha : float = kwargs['alpha']
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
    def __init__(self, **kwargs) -> None:
        super().__init__()
        alpha : float = kwargs['alpha']
        binary : bool = kwargs['binary']
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
        if self.binary:
            target = target.float()
        else:
            target = target.long()
            
        train_loss = self.train_loss(input, target)
        reg_term = self.aggr_function(**kwargs)
        return train_loss + self.alpha * reg_term
    
class DynamicSCFERegularizationLoss(Module):
    def __init__(self, initial_alpha: float, final_alpha: float, binary: bool, warmup_steps: int = 0, **kwargs) -> None:
        super().__init__()
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.binary = binary
        
        # New parameter: Number of steps to wait before ramping alpha
        self.warmup_steps = warmup_steps
        
        # Initialize the aggregation function using any extra kwargs
        self.aggr_function = get_aggr_func(**kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor, current_step: int = 0, num_steps: int = 1, **kwargs):
        """
        input: model's predictions (logits)
        target: true classes
        current_step: current batch step or epoch
        num_steps: total steps or epochs
        """
        
        # 1. Delayed Warm-up Alpha Interpolation
        if current_step < self.warmup_steps:
            # Phase 1: Pure baseline training (alpha remains at initial)
            alpha = self.initial_alpha
        else:
            # Phase 2: Active interpolation over the remaining steps
            active_step = current_step - self.warmup_steps
            active_total = num_steps - self.warmup_steps
            
            if active_total > 0:
                # min(1.0) strictly bounds the alpha from exceeding final_alpha
                progress = min(1.0, active_step / active_total)
                alpha = self.initial_alpha + progress * (self.final_alpha - self.initial_alpha)
            else:
                alpha = self.final_alpha

        # 2. Shape and Type Safety
        assert input.shape[0] == target.shape[0], f"Batch size mismatch: {input.shape[0]} vs {target.shape[0]}"
        
        if self.binary:
            train_loss = torch.nn.functional.binary_cross_entropy_with_logits(input.squeeze(), target.float().squeeze())
        else:
            assert input.dim() == 2, "Multiclass input must be of shape [N, C]"
            train_loss = torch.nn.functional.cross_entropy(input, target.long())
            
        # 3. Regularization Term
        reg_term = self.aggr_function(input=input, target=target, current_step=current_step, num_steps=num_steps, **kwargs)
        
        # Explicit consistency check: Ensure reg_term is a scalar
        if reg_term.dim() != 0:
            reg_term = reg_term.mean()

        # 4. Total Loss
        # NOTE: Using addition (+) because alpha is passed as a negative value in the config
        return train_loss + (alpha * reg_term)


    

class L1CrossEntropy(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        alpha : float = kwargs['alpha']
        binary : bool = kwargs['binary']
        self.binary = binary
        self.alpha = alpha
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
        weights: torch.Tensor = kwargs['weights']
        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1,"Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"

        if self.binary:
            target = target.float()
        else:
            target = target.long()
            
        train_loss = self.train_loss(input, target)
        l1_reg = sum(param.abs().sum() for param in weights)
        train_loss += self.alpha * l1_reg
        return train_loss
    
class L2CrossEntropy(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        alpha : float = kwargs['alpha']
        binary : bool = kwargs['binary']
        self.binary = binary
        self.alpha = alpha
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
        weights: torch.Tensor = kwargs['weights']
        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1,"Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"

        if self.binary:
            target = target.float()
        else:
            target = target.long()
            
        train_loss = self.train_loss(input, target)
        l2_reg = sum(param.pow(2).sum() for param in weights)
        return train_loss + self.alpha * l2_reg
    
# regterm = 1/distance + e #issue: 1/x^2 <- derivative 
class SCFEInverseRegularizationLoss(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        # NOTE: alpha must be POSITIVE! 
        self.alpha : float = kwargs['alpha']
        
        self.epsilon : float = kwargs.get('epsilon', 1e-6) 
        
        self.binary : bool = kwargs['binary']
        if self.binary: 
            self.train_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.train_loss = torch.nn.functional.cross_entropy
            
        self.aggr_function = get_aggr_func(**kwargs)

    def forward(self, **kwargs):
        """ input : model's predictions
            target: true classes
        """
        input : torch.Tensor = kwargs['input']
        target : torch.Tensor = kwargs['target']

        assert input.shape[0] == target.shape[0], "Batch size mismatch"
        if self.binary:
            assert input.dim() == 1, "Input must be of shape [N C]"
        else:
            assert input.dim() == 2, "Input must be of shape [N C]"
            
        if self.binary:
            target = target.float()
        else:
            target = target.long()
            
        # 1. Calculate Standard Cross-Entropy Loss
        train_loss = self.train_loss(input, target)
        
        # 2. Get the raw CF-Reg distance from your aggregation function
        distance = self.aggr_function(**kwargs)
        
        # 3. Apply the Inverse Penalty (Magnetic Repulsion)
        # As distance approaches 0, this fraction explodes, punishing the model.
        # As distance grows large, this fraction shrinks towards 0.
        penalty = 1.0 / (distance + self.epsilon)
        
        # Average the penalty if your aggregation function returns a vector
        if penalty.dim() > 0:
            penalty = penalty.mean()
            
        # 4. Add the penalty to the total loss
        return train_loss + (self.alpha * penalty)

class CrossEntropy(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        binary : bool = kwargs['binary']
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
        
        if self.binary:
            target = target.float()
        else:
            target = target.long()
            
        return self.train_loss(input, target)
    