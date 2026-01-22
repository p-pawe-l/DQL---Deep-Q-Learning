import abc
import numpy as np
import Interfaces.DeepLearning.FunctionInterface as fi
import Interfaces.DeepLearning.InitializerInterface as ii


"""Simple layer interface as building component for neural network"""
class Layer_Interface(abc.ABC):
        def __init__(self, 
                     input_units: int, 
                     output_units: int, 
                     activation_function: fi.Function_Interface,
                     weights_initializer: ii.Initializer_Interface,
                     biases_initializer: ii.Initializer_Interface, *args, **kwargs) -> None:
                self._input_units: int = input_units
                self._output_units: int = output_units 
                self._activation_function: fi.Function_Interface = activation_function
                
                self._weights: np.ndarray = weights_initializer.initialize(shape=(input_units, output_units), *args, **kwargs)
                self._biases: np.ndarray = biases_initializer.initialize(shape=(1, output_units), *args, **kwargs)
                
        @property
        def INPUT_UNITS(self) -> int: return self._input_units
        @property
        def OUTPUT_UNITS(self) -> int: return self._output_units
        
        @abc.abstractmethod
        def forward_pass(self, input: np.ndarray) -> np.ndarray:
                """Implements mathematical concept of forward pass in our neural network"""
                raise NotImplementedError
        
        @abc.abstractmethod
        def backprop_pass(self, output_gradient: np.ndarray) -> np.ndarray:
                """Implemented mathematical concept of backpropaagation pass in our neural network"""
                raise NotImplementedError 