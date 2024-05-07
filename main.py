import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from src.models.models import MLP
from src.trainer.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.plots.plots import plot_metrics
import numpy as np

np.random.seed(42)  # Setting seed for NumPy's RNG
torch.manual_seed(42)

# Additional steps to enforce determinism
# Note: These settings can degrade performance and may not guarantee complete reproducibility across different PyTorch releases or different platforms like CPUs and GPUs.

# Ensuring that all operations are deterministic on GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dtype = torch.float32

df=pd.read_csv('data/water_potability.csv')
df['ph'].fillna(value=df['ph'].median(),inplace=True)
df['Sulfate'].fillna(value=df['Sulfate'].median(),inplace=True)
df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median(),inplace=True)
X = df.drop('Potability',axis=1).values
y = df['Potability'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Parameters
input_dim = 9  # e.g., number of features in your input dataset
hidden_layers = [30, 20, 15, 5]  # e.g., two hidden layers with 50 and 30 neurons respectively
output_dim = 2  # e.g., number of classes for classification

# Create the MLP model
model = MLP(input_dim, hidden_layers, output_dim)

# Assuming model, criterion, optimizer have been defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()  # For classification tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_set = TensorDataset(torch.tensor(X_train, dtype=dtype), torch.tensor(y_train,dtype=torch.long))
test_set = TensorDataset(torch.tensor(X_test, dtype=dtype), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10)

# Initialize Trainer
trainer = Trainer(model, criterion, optimizer, device)
trainer.train(train_loader, test_loader, train_set, epochs=100)


plot_metrics(train_acc=trainer.train_acc_history,
             test_acc=trainer.test_acc_history,
             test_loss=trainer.test_loss_history,
             train_loss=trainer.train_loss_history,
             volume=trainer.p_x,
             volume_std=trainer.std)
