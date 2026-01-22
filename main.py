import multiprocessing
import os
import time
import gymnasium as gym
import minigrid
import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt 

import ML_algorithms as ml
import utils.preprocessing as pre
import utils.heurestics as hur
from utils.experiment_tracker import ExperimentTracker
from modules.Factory.ComponentFactory import ComponentFactory
from modules.Factory.ModelFactory import ModelFactory

ENV_EMPTY_8x8 = "MiniGrid-Empty-8x8-v0"           
ENV_EMPTY_5x5 = "MiniGrid-Empty-5x5-v0"           
ENV_EMPTY_16x16 = "MiniGrid-Empty-16x16-v0"       
ENV_EMPTY_RANDOM = "MiniGrid-Empty-Random-6x6-v0" 

ENV_FOUR_ROOMS = "MiniGrid-FourRooms-v0"         
ENV_SIMPLE_CROSSING = "MiniGrid-SimpleCrossing-S9N1-v0"

ENV_LAVA_GAP = "MiniGrid-LavaGapS5-v0"           
ENV_DOOR_KEY = "MiniGrid-DoorKey-5x5-v0"          

ENV_PLAYGROUND = "MiniGrid-Playground-v0"

ENV_LUNAR_LANDER = "LunarLander-v3"

ENV_NAME = ENV_FOUR_ROOMS

STATE_DIM = 7 * 7 * 3
ACTIONS_DIM = 3

EPISODES = 2000  
DECAY_RATE = 0.998
EPSYLON_MIN = 0.1 
EPSYLON_START = 1.0 
DISCOUNT_FACTOR = 0.5
LR=0.0005

TRAIN_NETWORK_FREQUENCY = 4
BATCH_SIZE = 32
MAX_REPLAY_BUFFER_SIZE = 20000


BASE_MODEL_PATH = 'Database/Models/Q_NET_147_128_64_7'

def plot_training_metrics(losses: list, rewards: list, successes: list, process_id: int):
        sb.set_theme(style="darkgrid", context="notebook", font_scale=1.1)
        
        def bin_data(data, bin_size):
            if not data:
                return np.array([]), np.array([]), np.array([])
            
            data = np.array(data)
            if len(data) < bin_size:
                return np.arange(len(data)), data, np.zeros(len(data))
                
            n_bins = len(data) // bin_size
            clean_len = n_bins * bin_size
            
            data_clean = data[:clean_len]
            data_reshaped = data_clean.reshape((n_bins, bin_size))
            
            means = np.mean(data_reshaped, axis=1)
            stds = np.std(data_reshaped, axis=1)
            
            x_axis = np.arange(bin_size/2, clean_len, bin_size)
            
            return x_axis, means, stds

        total_points = len(losses)
        bin_size = max(1, total_points // 50)

        plt.figure(figsize=(12, 6))
        x, y_mean, y_std = bin_data(losses, bin_size)
        
        if len(x) > 0:
            sb.lineplot(x=x, y=y_mean, label=f'Avg Loss (bin={bin_size})', linewidth=2.5)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        
        plt.title(f"Training Loss Trend (Process {process_id})", fontsize=16, fontweight='bold')
        plt.xlabel("Episode (Binned)", fontsize=12)
        plt.ylabel("Mean Loss", fontsize=12)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'Database/Images/loss_chart_proc_{process_id}.png', dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        x, y_mean, y_std = bin_data(rewards, bin_size)
        
        if len(x) > 0:
            sb.lineplot(x=x, y=y_mean, label=f'Avg Reward (bin={bin_size})', color='green', linewidth=2.5)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='green', alpha=0.2)
        
        plt.title(f"Training Reward Trend (Process {process_id})", fontsize=16, fontweight='bold')
        plt.xlabel("Episode (Binned)", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'Database/Images/rewards_chart_proc_{process_id}.png', dpi=300)
        plt.close()
        plt.figure(figsize=(12, 6))
        if len(successes) > 0:
            window_size = 100
            win_rates = []
            episodes = []

            for i in range(len(successes)):
                if i >= window_size - 1:
                    win_rate = np.mean(successes[i - window_size + 1:i + 1]) * 100
                    win_rates.append(win_rate)
                    episodes.append(i)

            if len(episodes) > 0:
                sb.lineplot(x=episodes, y=win_rates, label=f'Win Rate (rolling 100 ep)', color='purple', linewidth=2.5)
                plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Threshold')

        plt.title(f"Win Rate Over Time (Process {process_id})", fontsize=16, fontweight='bold')
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Win Rate (%)", fontsize=12)
        plt.ylim(0, 105)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'Database/Images/winrate_chart_proc_{process_id}.png', dpi=300)
        plt.close()

def train_worker(process_id):
        env = gym.make(ENV_NAME, render_mode=None)

        dqn_architecture = [
                {
                        'type': 'dense',
                        'neurons': 128,
                        'activation': 'relu',
                        'initializer': 'he_normal'
                },
                {
                        'type': 'dense',
                        'neurons': 64,
                        'activation': 'relu',
                        'initializer': 'he_normal'
                },
                {
                        'type': 'dense',
                        'neurons': ACTIONS_DIM,
                        'activation': 'linear',
                        'initializer': 'he_normal'
                }
        ]

        optimizer_config = {
                'name': 'adam',
        }

        experiment_config = {
                'env_name': ENV_NAME,
                'state_dim': STATE_DIM,
                'actions_dim': ACTIONS_DIM,
                'episodes': EPISODES,
                'lr': LR,
                'discount_factor': DISCOUNT_FACTOR,
                'epsilon_start': EPSYLON_START,
                'epsilon_min': EPSYLON_MIN,
                'decay_rate': DECAY_RATE,
                'batch_size': BATCH_SIZE,
                'max_replay_buffer_size': MAX_REPLAY_BUFFER_SIZE,
                'train_network_frequency': TRAIN_NETWORK_FREQUENCY,
                'architecture': dqn_architecture,
                'optimizer': optimizer_config['name'],
                'loss_function': 'huber'
        }
        tracker = ExperimentTracker(f'FourRooms_DQN_{EPISODES}ep', experiment_config)

        try:
                q_network = ModelFactory.get_model(
                        'q_deep_network',
                        STATE_DIM,
                        dqn_architecture,
                        optimizer_config,
                        loss_function='huber',
                        lr=LR
                )
        except Exception as e:
                print(f"[Process {process_id}] Error creating model: {e}")
                return

        agent = ml.Deep_QLearning(
                epsylon=1.0, 
                discount_factor=DISCOUNT_FACTOR,
                train_network_frequency=TRAIN_NETWORK_FREQUENCY,
                batch_size=BATCH_SIZE,
                max_buffer_size=MAX_REPLAY_BUFFER_SIZE, 
                actions=list(range(ACTIONS_DIM))
        )
        agent.set_model(q_network)

        mean_loss = []
        rewards_in_episode = []
        successes = []  
        loss_in_episode = 0
        
        grid = env.unwrapped.grid
        w = env.unwrapped.width
        h = env.unwrapped.height
        
        curriosty_table = np.ones((w, h), dtype=np.float32)

        for e in range(EPISODES):
                raw_obs, _ = env.reset()
                state = pre.preprocess_minigrid(raw_obs)

                done = False
                total_reward = 0
                reached_goal = False
                prev_pos = None
                current_pos = None
                step_count = 0

                while not done:
                        action = agent.produce_action(state)
                        
                        next_raw_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        if reward > 0:
                                reached_goal = True

                        next_state = pre.preprocess_minigrid(next_raw_obs)
                        current_pos = env.unwrapped.agent_pos
                        
                        #real_reward = hur.MINI_GRID_HEURESTIC_LAVA(reward, done)
                        real_reward = hur.MINI_GRID_HEURISTIC_FOURROOMS(reward, prev_pos, current_pos, 
                                                                        curriosty_table, e, step_count)
                        
                        prev_pos = current_pos
                        step_count += 1
                        agent.remember(state, action, real_reward, next_state, int(done))
                        
                        td_error_2 = agent._train_q_network(verbose=False)
                        
                        state = next_state
                        total_reward += real_reward
                        
                        if isinstance(td_error_2, np.ndarray):
                                loss_in_episode += np.mean(td_error_2)
                        elif isinstance(td_error_2, float):
                                loss_in_episode += td_error_2

                mean_loss.append(loss_in_episode)
                rewards_in_episode.append(total_reward)
                successes.append(1 if reached_goal else 0)

                tracker.log_episode(loss_in_episode, total_reward, reached_goal, agent._epsylon)

                loss_in_episode = 0

                agent.decay_epsilon_DECAY(decay_rate=DECAY_RATE, min_epsilon=EPSYLON_MIN)

                if e >= 99:
                        win_rate = np.mean(successes[-100:]) * 100
                else:
                        win_rate = np.mean(successes) * 100

                if e % 20 == 0:
                        print(f"[Process {process_id}] Episode {e}/{EPISODES}, Score: {total_reward:.2f}, Epsilon: {agent._epsylon:.2f}, Win Rate: {win_rate:.1f}%")

        save_path = f"{BASE_MODEL_PATH}_proc_{process_id}.pkl"
        print(f"[Process {process_id}] Saving model to {save_path}...")

        agent._q_network.save_model(save_path)

        os.makedirs('Database/Experiments', exist_ok=True)
        tracker.save(process_id)

        plot_training_metrics(mean_loss, rewards_in_episode, successes, process_id)

        final_win_rate = np.mean(successes[-100:]) * 100
        print(f"[Process {process_id}] Finished. Win Rate (last 100): {final_win_rate:.1f}%, Avg. score last 10: {np.mean(rewards_in_episode[-10:]):.2f}")
        env.close()

def run_parallel_training():
        num_processes = min(4, multiprocessing.cpu_count())        
        processes = []
        
        for i in range(num_processes):
                p = multiprocessing.Process(target=train_worker, args=(i,))
                processes.append(p)
                p.start()
        
        for p in processes:
                p.join()
                

def test_worker(process_id, episodes=5):
        model_path = f"{BASE_MODEL_PATH}_proc_{process_id}.pkl"
        if not os.path.exists(model_path):
            print(f"[Process {process_id}] Model not found: {model_path}")
            return

        env = gym.make(ENV_NAME, render_mode="human")
        
        try:
             q_network = ModelFactory.load_model('q_deep_network', model_path)
        except Exception as e:
             print(f"[Process {process_id}] Error loading model: {e}")
             return

        agent = ml.Deep_QLearning(
                epsylon=0.0,
                discount_factor=DISCOUNT_FACTOR,
                train_network_frequency=TRAIN_NETWORK_FREQUENCY,
                batch_size=BATCH_SIZE,
                max_buffer_size=MAX_REPLAY_BUFFER_SIZE,
                actions=list(range(ACTIONS_DIM))
        )
        agent.set_model(q_network)

        for e in range(episodes):
                raw_obs, _ = env.reset()
                state = pre.preprocess_minigrid(raw_obs)
                done = False
                total_reward = 0

                while not done:
                        action = agent.produce_action(state)
                        next_raw_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        next_state = pre.preprocess_minigrid(next_raw_obs)
                        total_reward += reward
                        state = next_state

                print(f"[Process {process_id}] Episode {e}: Reward: {total_reward}")
        
        env.close()

def run_parallel_testing():
        num_processes = 4        
        processes = []
        
        for i in range(num_processes):
                p = multiprocessing.Process(target=test_worker, args=(i, 10))
                processes.append(p)
                p.start()
        
        for p in processes:
                p.join()

if __name__ == "__main__":
        try:
                multiprocessing.set_start_method('spawn')
        except RuntimeError:
                pass
                
        start_time = time.time()
        
        os.makedirs('Database/Models', exist_ok=True)
        os.makedirs('Database/Images', exist_ok=True)
        
        run_parallel_training()
        run_parallel_testing()
        