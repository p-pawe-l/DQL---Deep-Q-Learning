import Interfaces.DeepLearning as IDL
import numpy as np
import pickle 


class Q_DeepNeuralNetwork(IDL.NeuralNetwork_Interface):
        def __init__(self, 
                     layers: list[IDL.Layer_Interface], 
                     loss_function: IDL.Function_Interface,
                     optimalizator: IDL.Optimizer_Interface,
                     lr: float = 0.01) -> None:
                super().__init__(layers, loss_function, optimalizator, lr)
        
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
                output_gradient: np.ndarray = self._loss_function.calculate_gradient(td_error) * -1
        
                for layer in reversed(self._layers):
                        output_gradient, weights_gradient, biases_gradient = layer.backprop_pass(output_gradient)
                        self._optimalizator.step(layer._weights, layer._biases, weights_gradient, biases_gradient, self._learning_rate)
                        
        """Works for recall batches"""
        def train_step(self, td_error: np.ndarray) -> np.ndarray:
                self._custom_back_propagation(td_error)
                
                # Returning squared td_error for logs
                return td_error ** 2
        
        def predict(self, input: np.ndarray) -> np.ndarray:
                predictions: np.ndarray = self._front_propagation(input)
                # Predictions is every Q(s(t), a(t)) value 
                return predictions
        
        """Saving model wiht pickle to filename"""
        def save_model(self, filename: str) -> None:
                with open(filename, 'wb') as file_handle:
                        checkpoints = {
                                'layers': self._layers,
                                'loss_function': self._loss_function,
                                'optimalizator': self._optimalizator,
                                'lr': self._learning_rate,
                        }
                        
                        pickle.dump(checkpoints, file_handle)
                        
        """Loading model with pickle from filename"""
        @staticmethod
        def load_model(filename: str) -> "Q_DeepNeuralNetwork":
                with open(filename, 'rb') as file_handle:
                        loaded_config = pickle.load(file_handle)
                        
                return Q_DeepNeuralNetwork(
                        layers=loaded_config['layers'],
                        loss_function=loaded_config['loss_function'],
                        optimalizator=loaded_config['optimalizator'],
                        lr=loaded_config['lr']
                )
                       
        
                       
                       
                
                        
                
        
        

