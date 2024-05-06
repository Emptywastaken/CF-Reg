import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from src.integration.montecarlo import MontecarloEstimator
from src.models.models import MLP
from src.trainer.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
dtype = torch.float32

df=pd.read_csv('data/water_potability.csv')
df['ph'].fillna(value=df['ph'].median(),inplace=True)
df['Sulfate'].fillna(value=df['Sulfate'].median(),inplace=True)
df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median(),inplace=True)
X = df.drop('Potability',axis=1).values
y = df['Potability'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

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

# Print the model structure
print(model)

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

trainer.train(train_loader, test_loader, train_set, epochs=1000)

plt.plot(trainer.p_x, label="P_x")
plt.plot(trainer.train_acc_history, label="Train Acc")
plt.plot(trainer.train_loss_history, label="Train Loss")


plt.legend()
plt.show()
