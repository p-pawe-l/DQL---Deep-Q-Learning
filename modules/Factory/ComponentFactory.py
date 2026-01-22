import Interfaces.DeepLearning as idl

from modules.Components.functions.activation_functions.linear import LinearFunction
from modules.Components.functions.activation_functions.relu import ReluFunction
from modules.Components.functions.loss_functions.huber_loss import HuberLossFunction
from modules.Components.functions.loss_functions.mse_loss import MSELossFunction
from modules.Components.functions.loss_functions.cross_entropy_loss import CrossEntropyLossFunction

from modules.Components.layers.DenseLayer import DenseLayer

from modules.Components.optimizers.SGDOptimizer import SGDOptimalizator
from modules.Components.optimizers.AdamOptimizer import AdamOptimizer

from modules.Components.initializers.HeNormalInitializer import HeNormalInitializer

class ComponentFactory:
        _functions: dict[str, type[idl.Function_Interface]] = {
                'linear': LinearFunction,
                'relu': ReluFunction,
                'huber': HuberLossFunction,
                'mse': MSELossFunction,
                'cross_entropy': CrossEntropyLossFunction
        }
        
        _layers: dict[str, type[idl.Layer_Interface]] = {
                'dense': DenseLayer
        }
        
        _optimizers: dict[str, type[idl.Optimizer_Interface]] = {
                'sgd': SGDOptimalizator,
                'adam': AdamOptimizer
        }
        
        _initializers: dict[str, type[idl.Initializer_Interface]] = {
                'he_normal': HeNormalInitializer
        }

        @staticmethod
        def get_function(name: str, *args, **kwargs) -> idl.Function_Interface:
                """Returns function class based on provided name"""
                if name not in ComponentFactory._functions:
                        raise ValueError(f"Function {name} not found")
                return ComponentFactory._functions[name](*args, **kwargs)
                
        @staticmethod
        def get_layer(name: str, *args, **kwargs) -> idl.Layer_Interface:
                """Returns layer class based on provided name"""
                if name not in ComponentFactory._layers:
                        raise ValueError(f"Layer {name} not found")
                return ComponentFactory._layers[name](*args, **kwargs)

        @staticmethod
        def get_optimizer(name: str, *args, **kwargs) -> idl.Optimizer_Interface:
                """Returns optimizer class based on provided name"""
                if name not in ComponentFactory._optimizers:
                        raise ValueError(f"Optimizer {name} not found")
                return ComponentFactory._optimizers[name](*args, **kwargs)
                
        @staticmethod
        def get_initializer(name: str, *args, **kwargs) -> idl.Initializer_Interface:
                """Returns initializer class based on provided name"""
                if name not in ComponentFactory._initializers:
                        raise ValueError(f"Initializer {name} not found")
                return ComponentFactory._initializers[name](*args, **kwargs)
