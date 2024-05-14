import torch
from src.integration.montecarlo import MontecarloEstimator

class RegularizedTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        self.p_x = []
        self.std = []

    def train_one_epoch(self, data_loader, m_e: MontecarloEstimator):
        
        total_loss = 0
        correct = 0
        total = 0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)
            out, target_cf = m_e.counterfactual(data, output)
            loss = self.criterion(output, target, out, target_cf)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            
        epoch_loss = total_loss / total
        epoch_accuracy = correct / total
        

        return epoch_loss, epoch_accuracy

    def train(self, train_loader, test_loader, train_set, epochs, wandb):

        self.model.train()
        m_e = MontecarloEstimator(self.model, train_set, n_samples=1000, radius=0.5)

        for epoch in range(epochs):
            
            epoch_loss, epoch_accuracy = self.train_one_epoch(train_loader, m_e)
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_accuracy)
            # Test the model after each training epoch
            test_loss, test_accuracy = self.test(test_loader, m_e)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_accuracy)
            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Total volume: {m_e.volume:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            wandb.log({"train_loss": epoch_loss, "test_loss": test_loss, "train_acc": epoch_accuracy, "test_acc": test_accuracy, "delta_loss": {torch.abs(epoch_loss - test_loss)}})
            
            
    def test(self, data_loader, m_e: MontecarloEstimator):

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                out, target_cf = m_e.counterfactual(data, output)
                loss = self.criterion(output, target, out, target_cf)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()
        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy
