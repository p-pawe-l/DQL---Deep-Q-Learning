import abc
import numpy as np


"""Interface for implementing Optimizer like Adam, AdaGrad, Stochastic etc."""
class Optimizer_Interface(abc.ABC):
        @abc.abstractmethod
        def step(self, 
                 weights: np.ndarray,
                 biases: np.ndarray, 
                 weights_gradient: np.ndarray, 
                 biases_gradient: np.ndarray, 
                 learning_rate: float, *args, **kwargs) -> None:
                """Implementation of step in gradient descent algorithm for our neural network"""
                raise NotImplementedError