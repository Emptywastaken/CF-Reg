from ..estimator import Estimator, SCFEEstimator, MontecarloEstimator, DiceEstimator, LatentSCFEEstimator

def get_estimator(**kwargs) -> Estimator:
    type : str = kwargs.pop("type")
    train_set = kwargs.pop("train_set")
    if type == "montecarlo":
        return MontecarloEstimator(function=kwargs.pop("function"),train_set = train_set, **kwargs)
    elif type == "scfe":
        return SCFEEstimator(function=kwargs.pop("function"), **kwargs) 
    elif type == "dice":
        return DiceEstimator(function=kwargs.pop("function"), **kwargs)
    elif type == "latent_scfe":
        return LatentSCFEEstimator(function=kwargs.pop("function"), **kwargs)
    else:
        raise ValueError(f"This estimator has not been implemented yet!")