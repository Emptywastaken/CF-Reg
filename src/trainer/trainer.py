import torch
from src.integration.montecarlo import MontecarloEstimator
import pytorch_lightning as L
from src.utility.evaluation import ClassifierEvaluator
from src.utility.optimizer import get_optimizer

class LightningClassifier(L.LightningModule):
    
    def __init__(self, model: torch.nn.Module, criterion, config, evaluator: ClassifierEvaluator) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.criterion = criterion
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.evaluator = evaluator
        
    def configure_optimizers(self):
        
        return get_optimizer(self.config.optimizer, self.model.parameters(), self.config.lr, l2=self.config.l2)
        
        
    def on_train_epoch_start(self) -> None:
        
        self.train_output = []
        self.train_target = []
        self.train_loss = []
    
    def on_train_epoch_end(self) -> None:
        
        stage: str = "train"
        accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
        
        self.log_dict({f"{stage}/loss": sum(self.train_loss)/len(self.train_loss), 
                       f"{stage}/accuracy": accuracy, 
                       f"{stage}/f1-score": f1, 
                       f"{stage}/precision": precision, 
                       f"{stage}/recall": recall}, on_epoch=True, on_step=False)  
        
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        
        self.train_target += target.tolist()
        self.train_output += output.tolist()
        self.train_loss += [loss.item()]
        
        return loss
    
    def on_validation_epoch_start(self) -> None:
        
        self.val_output = []
        self.val_target = []
        self.val_loss = []
    
    def on_validation_epoch_end(self) -> None:
        
        if self.trainer.state.stage != "sanity_check":
            
            stage: str = "validation"
            accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.val_output, self.val_target)
            
            self.log_dict({f"{stage}/loss": sum(self.val_loss)/len(self.val_loss), 
                        f"{stage}/accuracy": accuracy, 
                        f"{stage}/f1-score": f1, 
                        f"{stage}/precision": precision, 
                        f"{stage}/recall": recall}, on_epoch=True, on_step=False) 

    def validation_step(self, batch, batch_idx):
        
        data, target = batch
        output = self.model(data)
        val_loss = self.criterion(output, target) 
                    
        self.val_target += target.tolist()
        self.val_output += output.tolist()
        self.val_loss += [val_loss.item()]   
        
        return val_loss
         

class CounterfactualLightningClassifier(L.LightningModule):
    
    def __init__(self, model: torch.nn.Module, criterion, config, evaluator: ClassifierEvaluator, estimator: MontecarloEstimator) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.criterion = criterion
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.evaluator = evaluator
        self.estimator = estimator
        
    def configure_optimizers(self):
        
        return get_optimizer(self.config.optimizer, self.model.parameters(), self.config.lr, l2=self.config.l2)
        
        
    def on_train_epoch_start(self) -> None:
        
        self.train_output = []
        self.train_target = []
        self.train_loss = []
    
    def on_train_epoch_end(self) -> None:
        
        stage: str = "train"
        accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
        
        self.log_dict({f"{stage}/loss": sum(self.train_loss)/len(self.train_loss), 
                       f"{stage}/accuracy": accuracy, 
                       f"{stage}/f1-score": f1, 
                       f"{stage}/precision": precision, 
                       f"{stage}/recall": recall}, on_epoch=True, on_step=False)  
        
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        
        data, target = batch
        output = self.model(data)
        out, target_cf = self.estimator.get_counterfactual(data, output)
        loss = self.criterion(output, target, out, target_cf)        
        self.train_target += target.tolist()
        self.train_output += output.tolist()
        self.train_loss += [loss.item()]
        
        return loss
    
    def on_validation_epoch_start(self) -> None:
        
        self.val_output = []
        self.val_target = []
        self.val_loss = []
    
    def on_validation_epoch_end(self) -> None:
        
        if self.trainer.state.stage != "sanity_check":
            
            stage: str = "validation"
            accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.val_output, self.val_target)
            
            self.log_dict({f"{stage}/loss": sum(self.val_loss)/len(self.val_loss), 
                        f"{stage}/accuracy": accuracy, 
                        f"{stage}/f1-score": f1, 
                        f"{stage}/precision": precision, 
                        f"{stage}/recall": recall}, on_epoch=True, on_step=False) 

    def validation_step(self, batch, batch_idx):
        
        data, target = batch
        output = self.model(data)
        out, target_cf = self.estimator.get_counterfactual(data, output)
        val_loss = self.criterion(output, target, out, target_cf)   
        
        self.val_target += target.tolist()
        self.val_output += output.tolist()
        self.val_loss += [val_loss.item()]   
        
        return val_loss



class Trainer:
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

    def train_one_epoch(self, data_loader, m_e):
        
        total_loss = 0
        correct = 0
        total = 0
        classifier_out = []
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            classifier_out.append(predicted)

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy
    
    def train(self, train_loader, test_loader, train_set, epochs, wandb):

        self.model.train()

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self.train_one_epoch(train_loader, None)
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_accuracy)

            # Test the model after each training epoch
            test_loss, test_accuracy = self.test(test_loader)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_accuracy)
            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            wandb.log({"train_loss": epoch_loss, "test_loss": test_loss, "train_acc": epoch_accuracy, "test_acc": test_accuracy})
            
            
    def test(self, data_loader):

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy
