def get_optimizer(name: str, params, lr: float, **kwargs):
    
    regularization = kwargs["l2"] if "l2" in kwargs else 0.0

    
    if name.lower() == "adam":
        from torch.optim import Adam
        
        return Adam(params=params, lr=lr, weight_decay=regularization)
    
    elif name.lower() == "sgd":
        from torch.optim import SGD
        
        return SGD(params=params, lr=lr, weight_decay=regularization)