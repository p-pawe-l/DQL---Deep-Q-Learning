import Interfaces.DeepLearning as IDL
import numpy as np 


class HuberLossFunction(IDL.Function_Interface):
        def __init__(self, delta: float = 1.0) -> None:
                self._delta = delta
        
        def calculate_value(self, x: np.ndarray) -> np.ndarray:
                condition = np.abs(x) <= self._delta
                squared_loss = 0.5 * (x ** 2)
                linear_loss = self._delta * (np.abs(x) - (0.5 * self._delta))
                return np.where(condition, squared_loss, linear_loss)

        def calculate_gradient(self, x: np.ndarray) -> np.ndarray:
                condition = np.abs(x) <= self._delta
                grad_squared = x
                grad_linear = self._delta * np.sign(x)
                return np.where(condition, grad_squared, grad_linear)
                