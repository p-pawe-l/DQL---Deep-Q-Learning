import gymnasium as gym
import minigrid
import modules.models.Q_DeepNetwork as nn
import ML_algorithms as ml
import utils.preprocess_minigrid as pre

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
state_dim = 7 * 7 * 3
action_dim = env.action_space.n

layer_config = [
    (state_dim, 'relu'),
    (64, 'relu'),
    (action_dim, 'linear')
]
my_network = nn.Q_DeepNeuralNetwork(layers=layer_config, lr=0.01)

agent = ml.Deep_QLearning(
    epsylon=1.0, 
    learning_rate=0.001, 
    discount_factor=0.99, 
    actions=list(range(action_dim))
)

agent.set_model(my_network)
episodes = 500

for e in range(episodes):
    raw_obs, _ = env.reset()
    state = pre.preprocess(raw_obs)
    
    done = False
    total_reward = 0
    
    while not done:
        action = agent.produce_action(state)
        
        next_raw_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = pre.preprocess(next_raw_obs)
        
        real_reward = reward if reward > 0 else -0.01 
        agent.remember(state, action, real_reward, next_state, int(done))
        
        agent._train_q_network(verbose=False)
        
        state = next_state
        total_reward += real_reward
        
    agent.decay_epsilon(decay_rate=0.95)
    print(f"Epizod {e}, Wynik: {total_reward:.2f}, Epsilon: {agent._epsylon:.2f}")