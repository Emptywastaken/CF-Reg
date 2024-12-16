from abc import ABC, abstractmethod

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