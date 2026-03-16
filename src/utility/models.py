import math
import torch
import torch.nn as nn

def apply_custom_norm_init(module: nn.Module, expected_norm: float):
    """
    Applies a uniform initialization to a linear layer to guarantee 
    a specific expected L2 norm for its weights.
    """
    if isinstance(module, nn.Linear):
        d = module.weight.size(1) # Number of input features
        a = expected_norm * math.sqrt(3.0 / d)
        
        # Apply the mathematically bounded uniform distribution
        with torch.no_grad():
            nn.init.uniform_(module.weight, a=-a, b=a)

def get_model(**kwargs) -> torch.nn.Module:
    
    
    config: dict = kwargs["config"]
    model_type: str = config.pop("model_type")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(model_type)
    if model_type == "MLP":
        from src.models.models import MLP
        config['input_dim'] = config['input_dim'][1]
        model = MLP(**config)
        
        return model.to(device)

    elif model_type == "BMLP":
        from src.models.models import BMLP
        config['input_dim'] = config['input_dim'][1]
        model = BMLP(**config)

        return model.to(device)
    
    elif model_type == "BMLPBatch":
        from src.models.models import BMLPBatch
        config['input_dim'] = config['input_dim'][1]
        model = BMLPBatch(**config)

        return model.to(device)
    
    elif model_type == "CNN":
        from src.models.models import CNN
        config['input_dim'] = config['input_dim'][2]
        model = CNN(dimension_input=config["input_dim"], 
                    classes=config["nclasses"], 
                    channel_input=config["channel_in"], 
                    channel_list=config["channel_list"], 
                    kernel_list=config["kernel_list"])
        
        return model.to(device)
    
    elif model_type == "BCNN":
        from src.models.models import BCNN
        config['input_dim'] = config['input_dim'][2]
        model = BCNN(dimension_input=config["input_dim"], 
                    classes=config["nclasses"], 
                    channel_input=config["channel_in"], 
                    channel_list=config["channel_list"], 
                    kernel_list=config["kernel_list"])
        
        return model.to(device)
    
    elif model_type =="LogisticRegression":
        from src.models.models import BLogisticRegression
        config['input_dim'] = config['input_dim'][1]
        model = BLogisticRegression(** config)

        return model.to(device)
    
    elif model_type == "BPreActResNet":
        from src.models.models import BPreActResNet, PreActBlock
        model = BPreActResNet(PreActBlock, [2,2,2,2], num_classes=1)

        return model.to(device)
    else:
        
        raise ValueError(f"{model_type} is not a valide model type!")

    