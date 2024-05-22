from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch

class ClassifierEvaluator:
    
    def __init__(self, classes: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accuracy = Accuracy(task="multiclass", num_classes=classes).to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=classes).to(self.device)
        self.precision = Precision(task="multiclass", average='macro', num_classes=classes).to(self.device)
        self.recall = Recall(task="multiclass", average='macro', num_classes=classes).to(self.device)
        
    
    def get_complete_evaluation(self, output, target):
        
        output = torch.tensor(output, device=self.device)
        target = torch.tensor(target, device=self.device)
        
        output = torch.argmax(output, dim=1)
        
        accuracy = self.accuracy(output, target)
        f1 = self.f1(output, target)
        precision = self.precision(output, target)
        recall = self.recall(output, target)
        
        return accuracy, f1, precision, recall