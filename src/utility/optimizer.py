def get_optimizer(name: str, params, lr: float, **kwargs):
    
    if name.lower() == "adam":
        from torch.optim import Adam
        
        return Adam(params=params, lr=lr)
    
    elif name.lower() == "sgd":
        from torch.optim import SGD
        
        return SGD(params=params, lr=lr)