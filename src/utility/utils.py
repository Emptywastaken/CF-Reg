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
    
    
    
def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)