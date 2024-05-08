import torch

def get_model(type: str, **kwargs) -> torch.nn.Module:
    
    if type == "MLP":
        from src.models.models import MLP
        
        input_dim = kwargs["input_dim"]
        hidden_layers = kwargs["hidden_layers"]   
        output_dim = kwargs["out_classes"]  
        model = MLP(input_dim, hidden_layers, output_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        return model
    
    
    else:
        
        raise ValueError(f"{type} is not a valide model type!")

    