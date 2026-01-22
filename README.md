# Deep Q-Learning for MiniGrid Environments

Implementation of Deep Q-Network (DQN) for solving MiniGrid navigation tasks with multiprocessing support.

## Environment

**MiniGrid-FourRooms-v0**: A partially observable grid environment where the agent must navigate through rooms connected by doorways to reach a goal position.

## Network Architecture

```
Input Layer:  147 features (flattened observation)
Hidden Layer: 128 neurons (ReLU)
Hidden Layer: 64 neurons (ReLU)
Output Layer: 3 actions (turn left, turn right, move forward)
```

**Optimizer**: Adam
**Loss Function**: Huber Loss

## Training Results

Training was conducted with 4 parallel processes over 2000 episodes each. Representative results from one process:

### Success Rate

![Win Rate](Database/Images/winrate_chart_proc_0.png)

### Rewards

![Rewards](Database/Images/rewards_chart_proc_0.png)

### Loss Function

![Loss](Database/Images/loss_chart_proc_0.png)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 2000 |
| Learning Rate | 0.0005 |
| Discount Factor (gamma) | 0.5 |
| Epsilon Start | 1.0 |
| Epsilon Min | 0.1 |
| Epsilon Decay | 0.998 |
| Batch Size | 32 |
| Replay Buffer Size | 20,000 |
| Training Frequency | Every 4 steps |
| Parallel Processes | 4 |

## Usage

### Training

```bash
python3 main.py
```

The experiment tracker automatically logs metrics (loss, rewards, success rate, epsilon) to `Database/Experiments/` during training.


## Key Observations

**50% Success Rate Barrier**: Current configuration struggles to consistently exceed 50% success rate. This is likely due to:

1. **Low Discount Factor (0.5)**: FourRooms requires planning 20-30 steps ahead. With gamma=0.5, rewards decay too quickly for effective long-term planning.

2. **Exploration Strategy**: Fast epsilon decay (0.998) may cause premature convergence to suboptimal policies.

3. **Environment Complexity**: Partial observability and sparse rewards make this a challenging task.

## Suggested Improvements

```python
DISCOUNT_FACTOR = 0.95  # Better for long-horizon tasks
DECAY_RATE = 0.999      # Slower exploration decay
EPSILON_MIN = 0.15      # Higher minimum exploration
EPISODES = 5000         # More training time
```

## Repository Structure

```
DeepReinforcementLearning/
├── main.py                                      # Main training script with multiprocessing
├── ML_algorithms.py                             # DQN implementation
├── view_plots.py                                # Plot viewer utility
│
├── Interfaces/                                  # Abstract base classes for components
│   ├── DeepLearning/
│   │   ├── FunctionInterface.py                # Base interface for activation/loss functions
│   │   ├── InitializerInterface.py             # Base interface for weight initializers
│   │   ├── LayerInterface.py                   # Base interface for network layers
│   │   ├── NeuralNetworkInterface.py           # Base interface for neural networks
│   │   └── OptimizerInterface.py               # Base interface for optimizers
│   └── ReinforcmentLearning/
│       └── RewardFunctionInterface.py          # Base interface for reward functions
│
├── modules/                                     # Concrete implementations
│   ├── Components/
│   │   ├── functions/
│   │   │   ├── activation_functions/
│   │   │   │   ├── relu.py                     # ReLU activation
│   │   │   │   ├── sigmoid.py                  # Sigmoid activation
│   │   │   │   ├── tanh.py                     # Tanh activation
│   │   │   │   └── linear.py                   # Linear activation
│   │   │   └── loss_functions/
│   │   │       ├── mse_loss.py                 # Mean Squared Error loss
│   │   │       ├── huber_loss.py               # Huber loss (used in DQN)
│   │   │       └── cross_entropy_loss.py       # Cross Entropy loss
│   │   ├── initializers/
│   │   │   └── HeNormalInitializer.py          # He Normal weight initialization
│   │   ├── layers/
│   │   │   └── DenseLayer.py                   # Fully connected layer
│   │   ├── models/
│   │   │   └── Q_DeepNetwork.py                # Deep Q-Network model
│   │   └── optimizers/
│   │       ├── AdamOptimizer.py                # Adam optimizer
│   │       └── SGDOptimizer.py                 # SGD optimizer
│   └── Factory/
│       ├── ComponentFactory.py                 # Factory for creating components
│       └── ModelFactory.py                     # Factory for creating models
│
├── utils/
│   ├── experiment_tracker.py                   # Automatic experiment tracking & logging
│   ├── preprocessing.py                        # State preprocessing utilities
│   └── heurestics.py                           # Reward shaping functions
│
├── Database/
│   ├── Images/                                 # Training charts and visualizations
│   ├── Models/                                 # Saved model weights
│   └── Experiments/                            # Experiment data (JSON logs)
│
└── docs/                                        # Documentation
    ├── Theory/                                 # Theoretical background
    └── Other/                                  # Additional documentation
```

## Development Status

**Note**: The Deep Learning Interfaces currently need more implementation for comprehensive testing:
- Additional activation functions (LeakyReLU, ELU, Softmax, etc.)
- More weight initializers (Xavier/Glorot, LeCun, etc.)
- Additional layer types (Convolutional, Dropout, Batch Normalization)
- More optimizer implementations (RMSProp, AdaGrad, etc.)
- Extended loss functions and test coverage

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- Gymnasium
- Gymnasium[box2d]
- MiniGrid
- NumPy
- Matplotlib
- Seaborn
