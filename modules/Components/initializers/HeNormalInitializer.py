import Interfaces.DeepLearning as IDL
import numpy as np


class HeNormalInitializer(IDL.Initializer_Interface):
        def initialize(self, shape: tuple[int, int]) -> np.ndarray:
                fan_in = shape[0]
                std_dev = np.sqrt(2.0 / fan_in)
                random_weights = np.random.randn(*shape)

                return random_weights * std_dev