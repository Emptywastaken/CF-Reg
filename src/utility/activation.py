import torch
from torch import nn

class BoundedLinearActivaton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.where(input <= 0, torch.tensor(0.0, device=input.device), 
                             torch.where(input < 1, input, torch.tensor(1.0, device=input.device)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        grad_input[input >= 1] = 0
        return grad_input

# Wrapper module to use the custom activation in nn.Sequential or other nn.Modules
class BLA(nn.Module):
    def forward(self, input):
        return BoundedLinearActivaton.apply(input)