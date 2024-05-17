import torch

from src.models.models import CNN

def get_model(type: str, **kwargs) -> torch.nn.Module:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if type == "MLP":
        from src.models.models import MLP
        
        input_dim = kwargs["input_dim"]
        hidden_layers = kwargs["hidden_layers"]   
        output_dim = kwargs["out_classes"]  
        model = MLP(input_dim, hidden_layers, output_dim)
        
        return model.to(device)
    
    elif type == "CNN":
        
        model = CNN()
        
        return model.to(device)
    
    else:
        
        raise ValueError(f"{type} is not a valide model type!")

    