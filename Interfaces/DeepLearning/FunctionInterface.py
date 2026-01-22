import abc
import numpy as np 


"""Interface for implementing functions used during back and front propagation"""
class Function_Interface(abc.ABC):
        @abc.abstractmethod
        def calculate_value(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
                """Implementation of calculating value in provided x"""
                raise NotImplementedError
        
        @abc.abstractmethod
        def calculate_gradient(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
                """Implemetation of calculating gradient in provided x"""
                raise NotImplementedError