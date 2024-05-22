import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import Tuple
import os

def get_dataset(name: str) -> Tuple[TensorDataset, TensorDataset]:
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    dtype = torch.float32
    seed = 42

    if name == "water":
        
        try:
            df=pd.read_csv('data/water_potability.csv')
            
        except Exception:
            
            raise ValueError(f"Water dataset is not inside the data folder! cwd {os.getcwd()}")
        
        df['ph'].fillna(value=df['ph'].median(),inplace=True)
        df['Sulfate'].fillna(value=df['Sulfate'].median(),inplace=True)
        df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median(),inplace=True)
        
        X = df.drop('Potability',axis=1).values
        y = df['Potability'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype), torch.tensor(y_train,dtype=torch.long))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype), torch.tensor(y_test, dtype=torch.long))
        
        return train_set, test_set
    
    elif name == "mnist":
        from torchvision import datasets
        from torchvision import transforms
        
        training_data =   datasets.MNIST("data", train=True, download=True,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

        test_data = datasets.MNIST('data', train=False, download=True,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        
        train_set = TensorDataset(training_data.data.type(torch.float).unsqueeze(1), training_data.targets)
        test_set = TensorDataset(test_data.data.type(torch.float).unsqueeze(1), test_data.targets)

        return train_set, test_set
        
    else:
        raise ValueError(f"Dataset {name} is not available!")
            
    
    
if __name__ == "__main__":
    
    
    train, test = get_dataset(name="mnist")