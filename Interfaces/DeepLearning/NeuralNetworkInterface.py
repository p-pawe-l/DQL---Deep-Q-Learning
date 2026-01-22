import abc
import numpy as np 
import Interfaces.DeepLearning.FunctionInterface as fi
import Interfaces.DeepLearning.OptimizerInterface as oi
import Interfaces.DeepLearning.LayerInterface as li
import utils.build as ub


"""Simple interface for building non complex neural networks"""
class NeuralNetwork_Interface(abc.ABC):
        def __init__(self, 
                     layers: list[li.Layer_Interface],
                     loss_function: fi.Function_Interface, 
                     optimalizator: oi.Optimizer_Interface,
                     learning_rate: float, *args, **kwargs) -> None:

                self._layers: list[li.Layer_Interface] = layers
                self._optimalizator: oi.Optimizer_Interface = optimalizator
                self._loss_function: fi.Function_Interface = loss_function
                self._learning_rate: float = learning_rate
                
        @abc.abstractmethod
        def _front_propagation(self, network_input: np.ndarray, *args, **kwargs) -> np.ndarray:
                """Implementation of front propagation algorithm"""
                ...
        
        @abc.abstractmethod
        def _back_propagation(self, network_output: np.ndarray, target: np.ndarray, *args, **kwargs) -> float:
                """Implementation of back propagation algorithm"""
                ...
        
        @abc.abstractmethod
        def fit(self, x: list[np.ndarray], y: list[np.ndarray], batch_size: int, epochs: int, lr: float, *args, **kwargs) -> None:
                """Implementation training loop for neural network"""
                ...
        
        @abc.abstractmethod
        def predict(self, input: np.ndarray) -> tuple[int, np.ndarray]:
                """Implemented Neural Network prediction logic"""
                ...
                
        @abc.abstractmethod
        def save_model(self, filename: str) -> None:
                """Implemented logic of saving neural network parameters to filename"""
                raise NotImplementedError
                
        @abc.abstractmethod
        def load_model(self, filename: str) -> None:
                """Implemented logic of loading neural network paramters from filename"""
                raise NotImplementedError
        