import interfaces.deep_learning as idl
import modules.functions.manager as func_man
import modules.functions.loss_functions.huber_loss as hl
import modules.optimizers.SGDOptimizer as sgd
import numpy as np 


class Q_DeepNeuralNetwork(idl.NeuralNetwork_Interface):
        def __init__(self, 
                     layers: list[tuple[int, str]], 
                     loss_function: idl.Function_Interface = hl.HuberLossFunction(),
                     function_manager: dict[str, idl.Function_Interface] = func_man.function_manager,
                     optimalizator: idl.Optimalizator_Interface = sgd.SGDOptimalizator(),
                     lr: float = 0.01) -> None:
                super().__init__(layers, loss_function, function_manager, optimalizator)
                self._lr = lr
        
        """Other implementaion of trainin is needed"""
        def fit(self, x, y, batch_size, epochs, lr, *args, **kwargs) -> None:
                raise NotImplementedError("No implementation for Q_DeepNeuralNetwork!")
        
        """Works for data batches"""
        def _front_propagation(self, network_input: np.ndarray, *args, **kwargs) -> np.ndarray:
                network_output: np.ndarray = network_input
                for layer in self._layers:
                       network_output = layer.forward_pass(network_output, *args, **kwargs)
                return network_output
        
        """Custom back propagation algorithm is needed for Q Deep Neural Network"""
        def _back_propagation(self, network_output, target, *args, **kwargs) -> float:
                raise NotImplementedError("No implementation for Q_DeepNeuralNetwork!")
        
        """Back propagation algortithm for Q Deep Neural Network"""
        """Custom one is better beacuse we only need td_error to be known"""
        def _custom_back_propagation(self, td_error: np.ndarray) -> None:
                output_gradient: np.ndarray = self._loss_function.calculate_gradient(td_error)
        
                for layer in reversed(self._layers):
                        output_gradient, weights_gradient, biases_gradient = layer.backprop_pass(output_gradient)
                        self._optimalizator.step(layer._weights, layer._biases, weights_gradient, biases_gradient, self._lr)
                        
        """Works for recall batches"""
        def train_step(self, td_error: np.ndarray):
                self._custom_back_propagation(td_error)
                
                # Returning squared td_error for logs
                return td_error ** 2
        
        def predict(self, input: np.ndarray) -> np.ndarray:
                predictions: np.ndarray = self._front_propagation(input)
                # Predictions is every Q(s(t), a(t)) value 
                return predictions
                
        
        

