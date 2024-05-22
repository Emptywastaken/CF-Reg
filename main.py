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
from src.sweep_configs.sweeps import sweep_configuration, nosweep
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import multiprocessing as mp

from src.utility.trainer import get_trainer

# Optionally set WANDB_MODE environment variable


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
 
    wandb_logger = WandbLogger(project=cfg.logger.project)
    #wandb.init(project="counterfactual_overfitting")
    run = wandb.init(project=cfg.logger.project, mode=cfg.logger.mode, config=nosweep)
    
    device = 'cuda' if torch.cuda.is_available() and cfg.device == "cuda" else 'cpu'
    np.random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    
    

    trainset, testset = get_dataset(name="mnist")
    input_dim = trainset.tensors[0].shape[1] # TODO: Rendi questo generico
    hidden_layers = [30, 20, 5]
    out_classes = 10
    
    model = get_model(type="CNN", input_dim=input_dim, hidden_layers=hidden_layers, out_classes=out_classes)
    typ = "normal"
    estimator = None if typ != "regularized" else MontecarloEstimator(function=model, train_set=trainset, n_samples=wandb.config.samples, radius=wandb.config.radius)
    
    criterion = get_loss(name=typ, alpha=wandb.config.alpha)
    evaluator = ClassifierEvaluator(classes=out_classes)
    clf = get_trainer(type=typ, model=model, criterion=criterion, evaluator=evaluator, estimator=estimator, config=wandb.config)
    
    
    #clf = LightningClassifier(model=model, criterion=criterion, config=wandb.config, evaluator=evaluator)

    train_loader = DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=wandb.config.batch_size)
    wandb_logger.watch(model, log='gradients', log_freq=100)
    #trainer = get_trainer(type="normal")(model=model, criterion=criterion, optimizer=optimizer, device=device)

    #trainer.train(train_loader, test_loader, trainset, epochs=wandb.config.epochs, wandb=wandb)
    trainer = pl.Trainer(enable_progress_bar=True, max_epochs=wandb.config.epochs, logger=wandb_logger, log_every_n_steps=1, num_sanity_val_steps=0, accelerator="gpu")
    trainer.fit(clf, train_loader, test_loader)

if __name__ == "__main__":
    
    main()

    #sweep_id = wandb.sweep(sweep=sweep_configuration, project='counterfactual_overfitting')

    #wandb.agent(sweep_id, function=main)

# TODO: Sistema gli step su wandb
# TODO: Aggiungi il trainer regolarizzato
