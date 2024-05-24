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
    
    