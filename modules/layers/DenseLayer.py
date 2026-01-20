import interfaces.deep_learning as idl
import numpy as np
import modules.functions.manager as func_man


class DenseLayer(idl.Layer_Interface):
        def __init__(self, input_units: int, output_units: int, activation_function: idl.Function_Interface) -> None:
                super().__init__(input_units, output_units, activation_function)    
                            
                self._layer_input: np.ndarray | None = None
                self._layer_output: np.ndarray | None = None
                
        def forward_pass(self, input: np.ndarray) -> np.ndarray:                
                self._layer_input = input
                non_acitvated_biased_output: np.ndarray = np.dot(input, self._weights)
                non_activated_output: np.ndarray = non_acitvated_biased_output + self._biases
                activated_output: np.ndarray = self._activation_function.calculate_value(non_activated_output)
                self._layer_output = activated_output
                
                return activated_output

        def backprop_pass(self, output_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                d_activation = self._activation_function.calculate_gradient(self._layer_output)
                local_gradient = output_gradient * d_activation
                
                weights_gradient = np.dot(self._layer_input.T, local_gradient)
                biases_gradient = np.sum(local_gradient, axis=0, keepdims=True)
                input_gradient = np.dot(local_gradient, self._weights.T)

                return input_gradient, weights_gradient, biases_gradient
                
                