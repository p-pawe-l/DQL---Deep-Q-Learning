import Interfaces.DeepLearning as IDL
from modules.Factory.ComponentFactory import ComponentFactory
from modules.Components.models.Q_DeepNetwork import Q_DeepNeuralNetwork

class ModelFactory:
        @staticmethod
        def _build_layers(input_size: int, architecture_config: list[dict]) -> list[IDL.Layer_Interface]:
                layers = []
                current_input_size = input_size

                for layer_cfg in architecture_config:
                        layer_type = layer_cfg.get('type', 'dense')
                        neurons = layer_cfg.get('neurons')
                        act_name = layer_cfg.get('activation', 'relu')
                        init_name = layer_cfg.get('initializer', 'he_normal')
                        
                        activation = ComponentFactory.get_function(act_name)
                        initializer = ComponentFactory.get_initializer(init_name)

                        layer = ComponentFactory.get_layer(
                                name=layer_type,
                                input_units=current_input_size,
                                output_units=neurons,
                                activation_function=activation,
                                weights_initializer=initializer,
                                biases_initializer=initializer
                        )

                        layers.append(layer)
                        current_input_size = neurons
                        
                return layers

        @staticmethod
        def get_model(name: str, 
                      input_size: int, 
                      architecture_config: list[dict], 
                      optimizer_config: dict, 
                      loss_function: str = "huber", 
                      lr: float = 0.01, **kwargs ) -> IDL.NeuralNetwork_Interface:
                match name:
                        case "q_deep_network":
                                layers_list = ModelFactory._build_layers(input_size, architecture_config)
                        
                                opt_name = optimizer_config.get('name', 'sgd')
                                opt_kwargs = {k: v for k, v in optimizer_config.items() if k != 'name'}
                                optimizer = ComponentFactory.get_optimizer(opt_name, **opt_kwargs)
                                loss_func = ComponentFactory.get_function(loss_function)

                                return Q_DeepNeuralNetwork(
                                        layers=layers_list,
                                        loss_function=loss_func,
                                        optimalizator=optimizer,
                                        lr=lr,
                                        **kwargs
                                )
                        
                        case _:
                                raise ValueError(f"Model {name} not found")
        
        @staticmethod
        def load_model(name: str, path: str) -> IDL.NeuralNetwork_Interface:
                match name:
                        case "q_deep_network":
                                return Q_DeepNeuralNetwork.load_model(path)
                        case _:
                                raise ValueError(f"Model type '{name}' is not supported for loading.")