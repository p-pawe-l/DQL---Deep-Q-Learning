import interfaces.deep_learning as idl


# Functions
from modules.functions.activation_functions.linear import LinearFunction
from modules.functions.activation_functions.relu import ReluFunction


function_manager: dict[str, idl.Function_Interface] = {
        'linear': LinearFunction,
        'relu': ReluFunction
}