import torch
from torch.utils.data import DataLoader, TensorDataset
from src.plots.plots import plot_metrics
import numpy as np
from typing import List, Tuple
from src.utility.dataset import get_dataset
from src.utility.models import get_model
from src.utility.loss import get_loss
import wandb
from src.sweep_configs.sweeps import sweep_configuration, nosweep
from src.utility.optimizer import get_optimizer
from src.utility.trainer import get_trainer
import os

# Optionally set WANDB_MODE environment variable

def main():

    #wandb.init(project="counterfactual_overfitting")
    wandb.init(project="counterfactual_overfitting", mode="offline", config=nosweep)
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainset, testset = get_dataset(name="mnist")
    
    #input_dim = trainset.tensors[0].shape[1] # TODO: Rendi questo generico
    hidden_layers = [30, 20, 5]
    out_classes = 2

    model = get_model(type="CNN", input_dim=2, hidden_layers=hidden_layers, out_classes=out_classes)
    criterion = get_loss(name="regularized", alpha=wandb.config.alpha)
    optimizer = get_optimizer(name="adam", params=model.parameters(), lr=wandb.config.lr)

    train_loader = DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=wandb.config.batch_size)

    trainer = get_trainer(type="regularized")(model=model, criterion=criterion, optimizer=optimizer, device=device)

    trainer.train(train_loader, test_loader, trainset, epochs=wandb.config.epochs, wandb=wandb)


if __name__ == "__main__":
    
    main()

#sweep_id = wandb.sweep(sweep=sweep_configuration, project='counterfactual_overfitting')

#wandb.agent(sweep_id, function=main)

    # plot_metrics(train_acc=trainer.train_acc_history,
    #             test_acc=trainer.test_acc_history,
    #             test_loss=trainer.test_loss_history,
    #             train_loss=trainer.train_loss_history,
    #             volume=trainer.p_x,
    #             volume_std=trainer.std)
