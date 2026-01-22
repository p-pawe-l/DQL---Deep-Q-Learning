import modules.Components.models.Q_DeepNetwork as nn
import Interfaces.ReinforcmentLearning as IRL
import numpy as np 
import random
import copy
from collections import deque

state = int
action = int

# replay = [current state, action performed, reward received, next state, is_done]
# USED TO TRAIN Q DEEP NETWORK 
replay = tuple[state, action, float, state, int]


"""Epsilon-greedy variant"""
class Deep_QLearning:
        def __init__(self, 
                     epsylon: float, 
                     discount_factor: float,
                     train_network_frequency: int,
                     batch_size: int,
                     max_buffer_size: int,
                     actions: list) -> None:
                self._epsylon: float = epsylon
                self._discount_factor: float = discount_factor
                
                self._actions: list = actions
                
                self._q_network: nn.Q_DeepNeuralNetwork | None = None
                self._target_network: nn.Q_DeepNeuralNetwork | None = None
                
                self._reward_function: IRL.RewardFunction_Interface | None = None
                self._replay_buffer: deque[replay] = deque(maxlen=max_buffer_size)
                self._batch_size: int = batch_size

                self._train_step_counter: int = 0
                self._train_frequency: int = train_network_frequency
                
        """Set custom deep neural network for algorithm"""        
        def set_model(self, model: nn.Q_DeepNeuralNetwork) -> None:
                self._q_network = model
                self._target_network = copy.deepcopy(model)
        
        """Sets custom reward function for algorithm"""
        def set_reward_function(self, reward_function: IRL.RewardFunction_Interface) -> None:
                self._reward_function = reward_function
                
        """Picking the best action that neural net produces or with probablity epsylon random action"""
        """Returns the index of best action to choose from self._actions"""
        def _policy(self, states: np.ndarray) -> int:
                if self._q_network is None: raise RuntimeError("Set model before deciding which action to choose")
                prediction = self._q_network.predict(states)
                
                action_count = prediction.shape[1] 
                if np.random.rand() < self._epsylon:
                        return np.random.randint(0, action_count)
                
                return np.argmax(prediction)
        
        """Agent receives reward based on action he choosed in provided state"""
        def _reward_for_batch(self, states: np.ndarray, actions: np.ndarray) -> float:
                if self._reward_function is None: raise RuntimeError("Set reward function before calculating an reward")
                return self._reward_function.calculate_reward(states, actions)
                
        """Calculates TD Error"""
        """For single state action state + 1"""
        def _td_error(self, state_t_0: state, state_t_1: state, action_t_0: action) -> float:
                if self._q_network is None: raise RuntimeError("Set model before deciding which action to choose")
                # Current predictions
                # Deep q network gives our agent predictions what action it needs to perform in a given state
                prediction_t_0 = self._q_network.predict(state_t_0)
                q_value_t_0 = prediction_t_0[action_t_0]
                
                """state_t_0 --next-state--> state_t_1"""
                # Bellman target gives us numeric value of expected value that we can have in a given state after making an action
                def _bellman_target(current_state: state, next_state: state, current_action: action) -> float:
                        # We get some reward after performing action on current state
                        reward = self._reward_for_batch(current_state, current_action)
                        # We are 'now' in next state after applying current action to current state
                        # Our deep q neural network gives us another predictions what to do in state t+1 
                        # We need to fetch the best option from next state that we can perform so our agent 
                        # knows that being in state t+0 gives us opportunity to go to state t+1 that can have an
                        # action that is highly rewarded
                        prediction_t_1 = self._q_network.predict(next_state)
                        max_t = np.max(prediction_t_1)
                        
                        return reward + self._discount_factor * max_t
                
                td_error = _bellman_target(state_t_0, state_t_1, action_t_0) - q_value_t_0
                return td_error
        
        """Calculating td_error for batch data"""
        def _td_error_batch(self, states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
                current_q_values: np.ndarray = self._q_network.predict(states)
                next_q_values: np.ndarray = self._target_network.predict(next_states)
                                    
                max_q_next_values: np.ndarray = np.max(next_q_values, axis=1)
                bellman_target_batch: np.ndarray = rewards + (self._discount_factor * max_q_next_values * (1 - dones))

                batch_indicies = np.arange(len(states))
                prediction_for_taken_action = current_q_values[batch_indicies, actions]
                
                errors = bellman_target_batch - prediction_for_taken_action
                
                td_error_matrix = np.zeros_like(current_q_values)
                td_error_matrix[batch_indicies, actions] = errors
                
                return td_error_matrix
                                             
        
        def _train_q_network(self, verbose: bool = True) -> np.ndarray:
                if len(self._replay_buffer) < self._batch_size: return np.array([0])
                
                """Helper function for getting batch"""
                def _get_batch(batch_size: int):
                        batch_real_size: int = min(len(self._replay_buffer), batch_size)
                        batch_data = random.sample(self._replay_buffer, batch_real_size)
                        states, actions, rewards, next_states, dones = zip(*batch_data)
                        return (
                                np.array(states),     
                                np.array(actions),      
                                np.array(rewards),      
                                np.array(next_states),  
                                np.array(dones)         
                        )       
                
                # Fetchind data about batch and training q network to adapt 
                states, actions, rewards, next_states, dones  = _get_batch(self._batch_size)  
                td_batch_error: np.ndarray = self._td_error_batch(states, next_states, actions, rewards, dones)
                squread_td_error_batch: np.ndarray = self._q_network.train_step(td_batch_error)
                
                if verbose:
                        print(f"Total loss: {np.sum(squread_td_error_batch)}")
                
                if self._train_step_counter % self._train_frequency == 0:
                        self.update_train_network()
                        
                return squread_td_error_batch
                        

        def update_train_network(self) -> None:
                self._target_network = copy.deepcopy(self._q_network)

        def produce_action(self, state: np.ndarray) -> int:
                if self._q_network is None: raise RuntimeError("Model not set")
                
                action_index = self._policy(state)
                return action_index

        def remember(self, state, action, reward, next_state, done):
                s = state[0] if state.ndim > 1 else state
                ns = next_state[0] if next_state.ndim > 1 else next_state

                self._replay_buffer.append((s, int(action), reward, ns, done))

        """Linear drop for epislon"""
        def decay_epsilon_LINEAR(self, episode, num_episodes):
                if episode < num_episodes * 0.5:
                        self._epsylon = 1.0
                else:
                        progress = (episode - num_episodes * 0.5) / (num_episodes * 0.5)
                        self._epsylon = max(0.05, 1.0 - progress)
                        
        """Decaying epsilon"""
        def decay_epsilon_DECAY(self, decay_rate=0.995, min_epsilon=0.01):
                if self._epsylon > min_epsilon:
                        self._epsylon *= decay_rate
                        
                        
                        
                
                
        
        
                              