import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple
from src.integration.montecarlo import MontecarloEstimator
from src.utility.dataset import get_dataset
from src.utility.evaluation import ClassifierEvaluator
from src.utility.models import get_model
from src.utility.loss import get_loss
import wandb
from src.sweep_configs.sweeps import sweep_configuration_dynamic_alpha, nosweep
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import multiprocessing as mp
import os
from src.utility.trainer import get_trainer
from lightning.pytorch.utilities import disable_possible_user_warnings

# ignore all warnings that could be false positives
disable_possible_user_warnings()
# Optionally set WANDB_MODE environment variable


#PARAMETRI HYDRA

#dataset = "water"
#out_classes = 2
#model_type = "MLP"
#typ = "regularized"
#loss_type = "dyn_regularized"
# enable_progress_bar=True
# accelerator="gpu"
# num_sanity_val_steps=0

#PARAMETRI SWEEP

# nosweep = {
#     "batch_size": 256,
#     "epochs": 1500,
#     "lr": 0.0001,
#     "radius":  1.5,
#     "samples": 1000,
#     "alpha": 0.5,
#     "optimizer": "adam",
#     "l2": 0.0
# }

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    
    torch.set_float32_matmul_precision('high')
    
    if wandb.run:
        wandb.finish()
        
    #run = wandb.init(project=cfg.logger.project, mode=cfg.logger.mode, config=nosweep)
    run = wandb.init(project=cfg.logger.project, mode=cfg.logger.mode)
    
    batch_size = wandb.config.batch_size
    n_samples = wandb.config.samples
    radius = wandb.config.radius
    epochs = wandb.config.epochs
    alpha = 0.0
    
    wandb_logger = WandbLogger(project=cfg.logger.project)
    #wandb.init(project="counterfactual_overfitting")
    
    device = 'cuda' if torch.cuda.is_available() and cfg.device == "cuda" else 'cpu'
    np.random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainset, testset = get_dataset(name=cfg.data.name)
    
    hidden_layers = [30, 20, 5]
    
    model = get_model(type=cfg.model_type, input_dim=cfg.data.input_dim, hidden_layers=hidden_layers, out_classes=cfg.data.nclasses)

    estimator = None if "regularized" not in cfg.loss_type else MontecarloEstimator(function=model, train_set=trainset, n_samples=n_samples, radius=radius)
    
    criterion = get_loss(name=cfg.loss_type, alpha=alpha)
    
    evaluator = ClassifierEvaluator(classes=cfg.data.nclasses)
    
    clf = get_trainer(type=cfg.loss_type, model=model, criterion=criterion, evaluator=evaluator, estimator=estimator, config=wandb.config)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size)
    
    wandb_logger.watch(model, log='gradients', log_freq=100)

    trainer = pl.Trainer(enable_progress_bar=cfg.trainer.enable_progress_bar, 
                         max_epochs=epochs, 
                         logger=wandb_logger, 
                         num_sanity_val_steps=cfg.trainer.num_sanity_val_step, 
                         accelerator=cfg.trainer.accelerator)
    
    trainer.fit(clf, train_loader, test_loader)
    
    

if __name__ == "__main__":
    
    #main()

    sweep_id = wandb.sweep(sweep=sweep_configuration_dynamic_alpha, project='counterfactual_overfitting')

    wandb.agent(sweep_id, function=main)
