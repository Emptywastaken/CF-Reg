import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple
from src.estimator import MontecarloEstimator
from src.trainer import LightningClassifier
from src.utility import get_dataset, get_model, get_loss, merge_hydra_wandb, ClassifierEvaluator, read_yaml
import wandb
from src.sweep_configs.sweeps import sweep_configuration_dynamic_alpha, sweep_configuration_normal, sweep_configuration_normal_no_l2, nosweep
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import disable_possible_user_warnings
disable_possible_user_warnings()


@hydra.main(version_base="1.3", config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig):
    
    # sweep_config = read_yaml(f'wandb_sweeps_configs/{cfg.config_name}.yaml')
    # sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.project)

            
    def train():
        
        with wandb.init(project=cfg.logger.project, mode=cfg.logger.mode)  as run: 


            merge_hydra_wandb(cfg, wandb.config)
            

            # To increase performances on CUDA 
            torch.set_float32_matmul_precision('high')
            

            estimator = None
            counterfactual: bool = True if cfg.loss.type != "normal" else False
            wandb_logger = WandbLogger(project=cfg.logger.project)
            
            np.random.seed(cfg.seed) 
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.seed)
                torch.cuda.manual_seed_all(cfg.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            trainset, testset = get_dataset(name=cfg.data.name) 
            
            model = get_model(config=OmegaConf.to_container(cfg.model) | {"input_dim": cfg.data.input_dim, "output_dim": cfg.data.nclasses})
            criterion = get_loss(**cfg.loss)

            if "regularized" in cfg.loss.type:
                
                estimator = MontecarloEstimator(function=model, train_set=trainset, **cfg.estimator)
            
            evaluator = ClassifierEvaluator(classes=cfg.data.nclasses)
            
            clf =  LightningClassifier(model=model, 
                                       criterion=criterion, 
                                       optim_config=OmegaConf.to_container(cfg.optimizer), 
                                       evaluator=evaluator, 
                                       estimator=estimator, 
                                       counterfactual=counterfactual)
                
            train_loader = DataLoader(trainset, **cfg.loader)
            test_loader = DataLoader(testset, **cfg.loader)
            
            wandb_logger.watch(model, log='gradients', log_freq=100)

            trainer = pl.Trainer(**cfg.trainer, logger=wandb_logger)
            
            trainer.fit(clf, train_loader, test_loader)
    
    
    if cfg.run_mode == 'sweep':

        sweep_config = read_yaml(f'wandb_sweeps_configs/{cfg.logger.config}.yaml')
        sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.logger.project)
        wandb.agent(sweep_id=sweep_id, function=train)
        
    elif cfg.run_mode == "run":
        train()
        
    else:
        
        raise ValueError(f"Values for run_mode can be sweep or run, you insert {cfg.run_mode}")

if __name__ == "__main__":
    
    main()

    # Load sweep configuration from YAML file

    

#TODO: Modifica il modulo lightning per permettere di far girare anche gli altri casi
#TODO: aggiungi regolarizzazione l1 e dropout
#TODO: aggiungi la funzione simil-relu alle immagini in greyscale
