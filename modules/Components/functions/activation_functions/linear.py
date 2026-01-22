import Interfaces.DeepLearning as IDL
import numpy as np 


class LinearFunction(IDL.Function_Interface):
        def __init__(self, a: float = 1, b: float = 0) -> None:
                self._a: float = a
                self._b: float = b
                
        def calculate_value(self, x: np.ndarray) -> np.ndarray:
                return self._a * x + self._b
        
        def calculate_gradient(self, x: np.ndarray) -> np.ndarray:
                return self._a * np.ones_like(x)                