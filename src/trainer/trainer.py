from torch import nn
import torch
from carla.recourse_methods import RecourseMethod
from carla import MLModel

class Trainer:

    def __init__(self, 
                 model: MLModel, 
                 cf_model: RecourseMethod, 
                 dataset, 
                 optimizer:str="sgd", 
                 loss:str="bce")->None:
        
        self.model = model
        self.cf_model = cf_model
        self.dataset = dataset
        self.optimizer = self._get_optimizer(optimizer)
        self.loss_fn = self._get_loss(loss)

    def train_step(self):

        # TODO: implements batch training
        # TODO: implement accuracy 

        for sample in self.dataset:
                    
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_fn(output, target)
            loss.backward()
            return loss

    def test_step(self):

        # TODO: implements batch training
        # TODO: implement accuracy 

        pass

    def _get_optimizer(self, 
                       model:nn.Module, 
                       optimizer:str)->torch.optim.Optimizer:
        
        if optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=0.001)
        elif optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=0.001)
        
    def _get_loss(self, 
                  loss:str)->torch.nn.Module:

        if loss == "bce":

            return torch.nn.BCELoss()