from abc import ABC, abstractmethod

class GenericFunction(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_value(self, x:float)->float:

        pass

    
