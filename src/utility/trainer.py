def get_trainer(type: str):
    
    if type == "regularized":
        from src.trainer.trainer_regularized import RegularizedTrainer

        return RegularizedTrainer

    
    elif type == "normal":
        from src.trainer.trainer import Trainer

        return Trainer
    
    