import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple
from src.integration.montecarlo import MontecarloEstimator
from src.trainer.trainer import LightningClassifier
from src.utility.dataset import get_dataset
from src.utility.evaluation import ClassifierEvaluator
from src.utility.models import get_model
from src.utility.loss import get_loss
import wandb
from src.sweep_configs.sweeps import sweep_configuration_dynamic_alpha, sweep_configuration_normal, sweep_configuration_normal_no_l2, nosweep
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import disable_possible_user_warnings

disable_possible_user_warnings()

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    
    
    if wandb.run:
        wandb.finish()
        
    run = wandb.init(project=cfg.logger.project, mode=cfg.logger.mode)    
    
    
    
    model_config: dict = {
        "model_type": "MLP",
        "input_dim": cfg.data.input_dim,
        "hidden_layers": [30, 20, 5], 
        "output_dim": cfg.data.nclasses,
        "dropout": 0.5
    }
   
    
    loss_config: dict = {
        "name": "normal",
        "alpha": 0.1
    }
    
    
    optimizer_config = {
        "name": "adam",
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "eps": 0.000001   
    }
    
    loader_config = {
        "batch_size": 32,
        "shuffle": True,
        
    }
    
    estimator_config = {
        "n_samples": 100,
        "radius": 1.5
    }

    # To increase performances on CUDA 
    torch.set_float32_matmul_precision('high')
    

    estimator = None
    wandb_logger = WandbLogger(project=cfg.logger.project)
    
    np.random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainset, testset = get_dataset(name=cfg.data.name) 
       
    model = get_model(config=model_config)
    criterion = get_loss(**loss_config)

    if "regularized" in cfg.loss_type:
        
        estimator = MontecarloEstimator(function=model, train_set=trainset, **estimator_config)
    
    evaluator = ClassifierEvaluator(classes=cfg.data.nclasses)
    clf =  LightningClassifier(model=model, criterion=criterion, config=optimizer_config, evaluator=evaluator, estimator=estimator)
        
    train_loader = DataLoader(trainset, **loader_config)
    test_loader = DataLoader(testset, **loader_config)
    
    wandb_logger.watch(model, log='gradients', log_freq=100)

    trainer = pl.Trainer(enable_progress_bar=cfg.trainer.enable_progress_bar, 
                         max_epochs=cfg.trainer.max_epochs, 
                         logger=wandb_logger, 
                         num_sanity_val_steps=cfg.trainer.num_sanity_val_step, 
                         accelerator=cfg.trainer.accelerator)
    
    trainer.fit(clf, train_loader, test_loader)
    
    

if __name__ == "__main__":
    
    #main()

    sweep_id = wandb.sweep(sweep=sweep_configuration_normal, project='counterfactual_overfitting')

    wandb.agent(sweep_id=sweep_id, function=main)
    

#TODO: Modifica il modulo lightning per permettere di far girare anche gli altri casi
#TODO: aggiungi regolarizzazione l1 e dropout
#TODO: aggiungi la funzione simil-relu alle immagini in greyscale
