import abc

"""Simple interface for reward funtion"""
class RewardFunction_Interface(abc.ABC):
        @abc.abstractmethod
        def calculate_reward(self, state_t, action_t, *args, **kwargs) -> float: 
                ...