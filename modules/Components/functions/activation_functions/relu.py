import Interfaces.DeepLearning as IDL
import numpy as np


class ReluFunction(IDL.Function_Interface):
        def calculate_value(self, x: np.ndarray) -> np.ndarray:
                return np.maximum(0, x)
        
        def calculate_gradient(self, x: np.ndarray) -> np.ndarray:
                # Optimized: single pass, no copy, uses boolean mask directly
                return (x > 0).astype(np.float32)