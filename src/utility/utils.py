from omegaconf import DictConfig
import yaml


def merge_dict(dict_1: dict, dict_2: dict):

    dict_1.update({key: dict_2[key] for key in dict_1 if key in dict_2})

def merge_hydra_wandb(cfg, wandb):
    
    for k, v in cfg.items():
        if type(v) == DictConfig:
              
            merge_dict(v, wandb)
    
    pass

def read_yaml(filename):
    
    with open(filename, 'r') as file:
        return yaml.safe_load(file)