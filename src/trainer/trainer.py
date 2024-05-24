import torch
from src.estimator.montecarlo import MontecarloEstimator
import pytorch_lightning as L
from src.utility.evaluation import ClassifierEvaluator
from src.utility.optimizer import get_optimizer

# class LightningClassifier(L.LightningModule):
    
#     def __init__(self, 
#                  model: torch.nn.Module, 
#                  criterion: torch.nn.Module, 
#                  config: dict, 
#                  evaluator: ClassifierEvaluator) -> None:
#         super().__init__()

#         self.model = model
#         self.config = config
#         self.criterion = criterion
#         self.train_output = []
#         self.train_target = []
#         self.train_loss = []
#         self.val_output = []
#         self.val_target = []
#         self.val_loss = []
#         self.evaluator = evaluator
        
#     def configure_optimizers(self):
        
#         return get_optimizer(params=self.model.parameters(), config=self.config)
        
        
#     def on_train_epoch_start(self) -> None:
        
#         self.train_output = []
#         self.train_target = []
#         self.train_loss = []
    
#     def on_train_epoch_end(self) -> None:
        
#         stage: str = "train"
#         accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
        
#         self.log_dict({f"{stage}/loss": sum(self.train_loss)/len(self.train_loss), 
#                        f"{stage}/accuracy": accuracy, 
#                        f"{stage}/f1-score": f1, 
#                        f"{stage}/precision": precision, 
#                        f"{stage}/recall": recall}, on_epoch=True, on_step=False)  
        
        
#     def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        
#         data, target = batch
#         output = self.model(data)
#         loss = self.criterion(output, target)
        
#         self.train_target += target.tolist()
#         self.train_output += output.tolist()
#         self.train_loss += [loss.item()]
        
#         return loss
    
#     def on_validation_epoch_start(self) -> None:
        
#         self.val_output = []
#         self.val_target = []
#         self.val_loss = []
    
#     def on_validation_epoch_end(self) -> None:
        
#         if self.trainer.state.stage != "sanity_check":
            
#             stage: str = "validation"
#             accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.val_output, self.val_target)
            
#             self.log_dict({f"{stage}/loss": sum(self.val_loss)/len(self.val_loss), 
#                         f"{stage}/accuracy": accuracy, 
#                         f"{stage}/f1-score": f1, 
#                         f"{stage}/precision": precision, 
#                         f"{stage}/recall": recall}, on_epoch=True, on_step=False) 

#     def validation_step(self, batch, batch_idx):
        
#         data, target = batch
#         output = self.model(data)
#         val_loss = self.criterion(output, target) 
                    
#         self.val_target += target.tolist()
#         self.val_output += output.tolist()
#         self.val_loss += [val_loss.item()]   
        
#         return val_loss
         

class LightningClassifier(L.LightningModule):
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module,
                 optim_config: dict,
                 evaluator: ClassifierEvaluator,
                 estimator: MontecarloEstimator,
                 counterfactual: bool) -> None:
        
        super().__init__()

        self.model = model
        self.optim_config = optim_config
        self.criterion = criterion
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.evaluator = evaluator
        self.estimator = estimator
        self.counterfactual = counterfactual
        
    def configure_optimizers(self):
        
        return get_optimizer(params=self.model.parameters(), config=self.optim_config)
        
        
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
        values: dict = {"input": output, "target": target}
        if self.counterfactual:
            out, target_cf = self.estimator.get_counterfactual(data, output)
            values = values | { "out": out, "target_cf": target_cf}
            
        loss = self.criterion(**values)        
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
        values: dict = {"input": output, "target": target}
        if self.counterfactual:
            out, target_cf = self.estimator.get_counterfactual(data, output)
            values = values | { "out": out, "target_cf": target_cf}        
            
        val_loss = self.criterion(**values)   
        self.val_target += target.tolist()
        self.val_output += output.tolist()
        self.val_loss += [val_loss.item()]   
        
        return val_loss
