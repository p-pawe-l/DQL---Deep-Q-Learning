import abc
import numpy as np
import utils.build as ub


"""Interface for implementing optimalizators like Adam, AdaGrad, Stochastic etc."""
class Optimalizator_Interface(abc.ABC):
        @abc.abstractmethod
        def step(weights: np.ndarray, biases: np.ndarray, weights_gradient: np.ndarray, biases_gradient: np.ndarray, learning_rate: float, *args, **kwargs) -> None:
                """Implementation of step in gradient descent algorithm for our neural network"""
                ...


"""Interface for implementing functions used during back and front propagation"""
class Function_Interface(abc.ABC):
        @abc.abstractmethod
        def calculate_value(x: np.ndarray) -> np.ndarray:
                """Implementation of calculating value in provided x"""
                ...
        
        @abc.abstractmethod
        def calculate_gradient(x: np.ndarray) -> np.ndarray:
                """Implemetation of calculating gradient in provided x"""
                ...


"""Simple layer interface as building component for neural network"""
class Layer_Interface(abc.ABC):
        def __init__(self, 
                     input_units: int, 
                     output_units: int, 
                     activation_function: Function_Interface) -> None:
                self._input_units: int = input_units
                self._output_units: int = output_units 
                self._activation_function: Function_Interface = activation_function
                
                self._weights: np.ndarray = np.random.randn(input_units, output_units) * 0.1
                self._biases: np.ndarray = np.zeros((1, output_units))
                
        @property
        def INPUT_UNITS(self) -> int: return self._input_units
        @property
        def OUTPUT_UNITS(self) -> int: return self._output_units
        
        @abc.abstractmethod
        def forward_pass(self, input: np.ndarray) -> np.ndarray:
                """Implements mathematical concept of forward pass in our neural network"""
                ...
        @abc.abstractmethod
        def backprop_pass(self, output_gradient: np.ndarray) -> np.ndarray:
                """Implemented mathematical concept of backpropaagation pass in our neural network"""
                
                
"""Simple interface for building non complex neural networks"""
class NeuralNetwork_Interface(abc.ABC):
        def __init__(self, 
                     layers: list[tuple[int, str]],
                     loss_function: Function_Interface, 
                     function_manager: dict[str, Function_Interface], 
                     optimalizator: Optimalizator_Interface) -> None:
                # Automatically building layers
                self._layers: list[Layer_Interface] = ub.build_layers(layers, function_manager)
                self._function_manager: dict[str, Function_Interface] = function_manager
                self._optimalizator: Optimalizator_Interface = optimalizator
                self._loss_function: Function_Interface = loss_function
                
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
        
        
        
                
                