import torch
from src.trainer.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from src.plots.plots import plot_metrics
import numpy as np
from typing import List, Tuple
from src.utility.dataset import get_dataset
from src.utility.models import get_model

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(seed)  # Setting seed for NumPy's RNG
torch.manual_seed(seed)

# Additional steps to enforce determinism
# Note: These settings can degrade performance and may not guarantee complete reproducibility across different PyTorch releases or different platforms like CPUs and GPUs.

# Ensuring that all operations are deterministic on GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


trainset, testset = get_dataset(name="water")

# Parameters
input_dim = trainset.tensors[0].shape[1]
hidden_layers = [30, 20, 5]
out_classes = 2

model = get_model(type="MLP", input_dim=input_dim, hidden_layers=hidden_layers, out_classes=out_classes)

criterion = torch.nn.CrossEntropyLoss()  # For classification tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=10)

# Initialize Trainer
trainer = Trainer(model, criterion, optimizer, device)
trainer.train(train_loader, test_loader, trainset, epochs=500)


plot_metrics(train_acc=trainer.train_acc_history,
             test_acc=trainer.test_acc_history,
             test_loss=trainer.test_loss_history,
             train_loss=trainer.train_loss_history,
             volume=trainer.p_x,
             volume_std=trainer.std)
