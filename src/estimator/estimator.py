from abc import ABC, abstractmethod
from torch import Tensor

class Estimator(ABC):
    """
    Abstract base class for Estimators.
    """

    @abstractmethod
    def get_estimate_name(self) -> str:
        """
        Abstract method to return the name of estimation.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def build_log(self, values: list, stage: str) -> dict:
        """
        Abstract method to build a dictionary of logs metrics starting from the results of measurements.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def get_estimate(self, data: Tensor, output: Tensor) -> Tensor:
        """
        Abstract method to build a tensor of norms of the difference between each sample in data and its counterfactual example.
        Subclasses must implement this method.
        """
        pass