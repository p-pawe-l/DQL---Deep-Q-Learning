def build_layers(layers_architecture: list[tuple[int, str]], function_manager: dict) -> list:
        from modules.layers.DenseLayer import DenseLayer
        layers = []
    
        for i in range(len(layers_architecture) - 1):
                input_units, _ = layers_architecture[i]
                
                output_units, activation_name = layers_architecture[i+1]
                
                activation_class = function_manager.get(activation_name)
                if activation_class is None:
                        raise ValueError(f"Activation function '{activation_name}' not found provided function manager.")
                
                activation_function = activation_class()
                layer = DenseLayer(input_units, output_units, activation_function)
                layers.append(layer)
                
        return layers