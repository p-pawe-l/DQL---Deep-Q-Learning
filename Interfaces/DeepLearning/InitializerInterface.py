import abc 
import numpy as np


"""Interface for implementing weight initialization strategies"""
class Initializer_Interface(abc.ABC):
        @abc.abstractmethod
        def initialize(self, shape: tuple[int, int], *args, **kwargs) -> np.ndarray:
                """Implementation for initializing array of gives shape"""
                raise NotImplementedError