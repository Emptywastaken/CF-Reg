from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch

class ClassifierEvaluator:
    
    def __init__(self, classes: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.accuracy = Accuracy(task="multiclass", num_classes=classes).to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=classes).to(self.device)
        self.precision = Precision(task="multiclass", average='macro', num_classes=classes).to(self.device)
        self.recall = Recall(task="multiclass", average='macro', num_classes=classes).to(self.device)
        self.crossentropy = torch.nn.functional.binary_cross_entropy_with_logits
        
    
    def get_complete_evaluation(self, output, target):
        
        output = torch.tensor(output, device=self.device)
        target = torch.tensor(target, device=self.device)

        crossentropy = self.crossentropy(output, target)
        
        output = torch.argmax(output, dim=-1) if output.ndim > 1 else (output > 0.5).float()
        
        accuracy = self.accuracy(output, target)
        f1 = self.f1(output, target)
        precision = self.precision(output, target)
        recall = self.recall(output, target)
       
        
        return accuracy, f1, precision, recall, crossentropy