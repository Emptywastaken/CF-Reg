from typing import Any, Dict
from omegaconf import DictConfig, DictKeyType, OmegaConf
import yaml


def merge_dict(dict_1: dict | DictConfig, dict_2: dict):

    dict_1.update({key: dict_2[key] for key in dict_1 if key in dict_2})

#def merge_hydra_wandb(cfg, wandb):
    
#    for k, v in cfg.items():

def merge_hydra_wandb(dict1, dict2):
    """
    Merge dict2 into dict1 based on the provided rules.
    """
    OmegaConf.set_struct(dict1, False)
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict | DictConfig) and isinstance(value, dict | DictConfig):
                # Both values are dictionaries; merge recursively
                merge_hydra_wandb(dict1[key], value)
            else:
                # Override the value in dict1 with the value from dict2
                dict1[key] = value
        else:
            if "." in key:
                path = key.split('.')
                t = dict1
                for p in path[:-1]:
                    t = t[p]
                t[path[-1]] = value
                continue
            if isinstance(value, dict| DictConfig):
                # Check if any subkey in dict2[key] is a key in dict1
                for subkey in value:
                    if subkey in dict1:
                        if isinstance(dict1[subkey], dict| DictConfig):
                            # Merge sub-dictionary
                            merge_hydra_wandb(dict1[subkey], value[subkey])
                        else:
                            # Override the value in dict1
                            dict1[subkey] = value[subkey]
                    else:
                        # Add the new subkey to dict1
                        dict1[subkey] = value[subkey]
            else:
                # Add the new key-value pair to dict1
                print(key, value)
                dict1[key] = value
    OmegaConf.set_struct(dict1, True)
    return dict1
    

def read_yaml(filename):
    
    with open(filename, 'r') as file:
        return yaml.safe_load(file)
    
    
    
def flatten_dict(d: Dict[Any, Any] | DictConfig, parent_key: str | DictKeyType = '', sep: str = '_') -> dict:
    
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)