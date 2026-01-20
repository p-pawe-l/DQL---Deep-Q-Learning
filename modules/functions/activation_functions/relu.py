import interfaces.deep_learning as idl
import numpy as np


class ReluFunction(idl.Function_Interface):
        def calculate_value(self, x: np.ndarray) -> np.ndarray:
                return np.maximum(0, x)
        
        def calculate_gradient(self, x: np.ndarray) -> np.ndarray:
                dx = x.copy()
                dx[dx <= 0] = 0
                dx[dx > 0] = 1
                
                return dx