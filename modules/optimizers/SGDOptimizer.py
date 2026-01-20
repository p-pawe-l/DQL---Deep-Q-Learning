import interfaces.deep_learning as idl
import numpy as np


class SGDOptimalizator(idl.Optimalizator_Interface):
        def step(self,
                 weights: np.ndarray, 
                 biases: np.ndarray, 
                 weights_gradient: np.ndarray, 
                 biases_gradient: np.ndarray, 
                 learning_rate: float, 
                 *args, 
                 **kwargs) -> None:
                
                weights_gradient = np.clip(weights_gradient, -10, 10)
                biases_gradient = np.clip(biases_gradient, -10, 10)
                
                weights -= learning_rate * weights_gradient
                biases -= learning_rate * biases_gradient
                