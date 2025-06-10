def get_trainer(type: str, model, criterion, evaluator, config,  estimator = None):
    
    # if "regularized" in type:
    #     from src.trainer.trainer import CounterfactualLightningClassifier

    #     return CounterfactualLightningClassifier(model=model,
    #                                              criterion=criterion,
    #                                              config=config,
    #                                              evaluator=evaluator,
    #                                              estimator=estimator)
    
    # elif type == "normal":
    from src.trainer.trainer import LightningClassifier

    return LightningClassifier(model=model,
                                   criterion=criterion,
                                   config=config,
                                   evaluator=evaluator)
    
def get_callbacks(
    early_stop_enable: bool = False,
    monitor: str = 'validation/loss',
    min_delta: float = 0.00,
    patience: int = 3,
    verbose: bool = True,
    mode: str = 'min'
    ):

    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    callbacks = []
    
    if early_stop_enable:
        early_stop = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode
        )
        callbacks.append(early_stop)
    
    return callbacks if len(callbacks) > 0 else None
    