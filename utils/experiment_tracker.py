import json
import numpy as np
from datetime import datetime
import os

class ExperimentTracker:
        def __init__(self, experiment_name: str, config: dict):
                self.experiment_name = experiment_name
                self.config = config
                self.start_time = datetime.now()
                self.metrics = {
                        'losses': [],
                        'rewards': [],
                        'successes': [],
                        'epsilon_values': []
                }

        def log_episode(self, loss: float, reward: float, success: bool, epsilon: float):
                self.metrics['losses'].append(float(loss))
                self.metrics['rewards'].append(float(reward))
                self.metrics['successes'].append(int(success))
                self.metrics['epsilon_values'].append(float(epsilon))

        def save(self, process_id: int):
                end_time = datetime.now()

                successes = np.array(self.metrics['successes'])
                rewards = np.array(self.metrics['rewards'])

                win_rates = []
                for i in range(len(successes)):
                        if i >= 99:
                                win_rate = np.mean(successes[i-99:i+1]) * 100
                                win_rates.append(float(win_rate))

                final_win_rate = np.mean(successes[-100:]) * 100 if len(successes) >= 100 else np.mean(successes) * 100
                max_win_rate = max(win_rates) if win_rates else final_win_rate

                data = {
                        'experiment_name': self.experiment_name,
                        'process_id': process_id,
                        'start_time': self.start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'duration_seconds': (end_time - self.start_time).total_seconds(),
                        'config': self.config,
                        'metrics': {
                        'losses': self.metrics['losses'],
                        'rewards': self.metrics['rewards'],
                        'successes': self.metrics['successes'],
                        'epsilon_values': self.metrics['epsilon_values']
                        },
                        'statistics': {
                        'total_episodes': len(self.metrics['successes']),
                        'final_win_rate': float(final_win_rate),
                        'max_win_rate': float(max_win_rate),
                        'avg_reward': float(np.mean(rewards)),
                        'final_100_avg_reward': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),
                        'total_successes': int(np.sum(successes)),
                        'avg_loss': float(np.mean(self.metrics['losses'])),
                        'final_epsilon': float(self.metrics['epsilon_values'][-1]) if self.metrics['epsilon_values'] else 0.0
                        }
                }

                os.makedirs('Database/Experiments', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'Database/Experiments/{self.experiment_name}_proc_{process_id}_{timestamp}.json'

                with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)

                return filename
