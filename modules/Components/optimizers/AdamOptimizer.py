import Interfaces.DeepLearning as IDL
import numpy as np


class AdamOptimizer(IDL.Optimizer_Interface):
        def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
                self.beta1 = beta1  
                self.beta2 = beta2
                self.epsilon = epsilon  

                # Per-layer state stored by id
                self._layer_states: dict[int, dict] = {}
                self._layer_timesteps: dict[int, int] = {} 
        
        def _get_layer_state(self, weights: np.ndarray, biases: np.ndarray) -> dict:
                layer_id = id(weights)

                if layer_id not in self._layer_states:
                        self._layer_states[layer_id] = {
                                'm_weights': np.zeros_like(weights),
                                'v_weights': np.zeros_like(weights),
                                'm_biases': np.zeros_like(biases),
                                'v_biases': np.zeros_like(biases),
                        }
                        self._layer_timesteps[layer_id] = 0

                return self._layer_states[layer_id]
        
        def step(self,
                 weights: np.ndarray, 
                 biases: np.ndarray, 
                 weights_gradient: np.ndarray, 
                 biases_gradient: np.ndarray, 
                 learning_rate: float, 
                 *args, 
                 **kwargs) -> None:
                
                weights_gradient = np.clip(weights_gradient, -1, 1)
                biases_gradient = np.clip(biases_gradient, -1, 1)

                state = self._get_layer_state(weights, biases)
                layer_id = id(weights)

                self._layer_timesteps[layer_id] += 1
                t = self._layer_timesteps[layer_id]

                state['m_weights'] = self.beta1 * state['m_weights'] + (1 - self.beta1) * weights_gradient
                state['m_biases'] = self.beta1 * state['m_biases'] + (1 - self.beta1) * biases_gradient

                state['v_weights'] = self.beta2 * state['v_weights'] + (1 - self.beta2) * (weights_gradient ** 2)
                state['v_biases'] = self.beta2 * state['v_biases'] + (1 - self.beta2) * (biases_gradient ** 2)

                m_weights_corrected = state['m_weights'] / (1 - self.beta1 ** t)
                m_biases_corrected = state['m_biases'] / (1 - self.beta1 ** t)
                v_weights_corrected = state['v_weights'] / (1 - self.beta2 ** t)
                v_biases_corrected = state['v_biases'] / (1 - self.beta2 ** t)
                
                weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
                biases -= learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)
        
        def reset(self) -> None:
                self._layer_states.clear()
                self._layer_timesteps.clear()
