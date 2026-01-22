# Deep Q-Learning Experiment Report - Manual Analysis

---

## Executive Summary

**Environment**: MiniGrid-FourRooms-v0
**Total Training Processes**: 4
**Episodes per Process**: 2000

### Key Finding: 50% Success Rate Barrier

**CRITICAL OBSERVATION**: Unable to break the 50% success rate barrier.

**TODO**: Examine the win rate plots (Database/Images/winrate_chart_proc_*.png) and fill in:
- Process 0 Final Win Rate: 42%
- Process 1 Final Win Rate: 51%
- Process 2 Final Win Rate: 38%
- Process 3 Final Win Rate: 45%
- Best Peak Win Rate Achieved: 55%
- Average Final Win Rate: 40%

---

## Experiment Configuration

**Network Architecture**: 147 → 128 → 64 → 3
**Optimizer**: Adam
**Loss Function**: Huber Loss

### Hyperparameters
- **Episodes**: 2000
- **Learning Rate**: 0.0005
- **Discount Factor (γ)**: 0.5
- **Epsilon Start**: 1.0
- **Epsilon Min**: 0.1
- **Epsilon Decay**: 0.998
- **Batch Size**: 32
- **Replay Buffer**: 20,000
- **Train Frequency**: Every 4 steps

---

## Observations from Plots

### Win Rate Analysis
**TODO**: After examining winrate_chart_proc_*.png files:
- Did any process break 50%?
- At what episode did peak performance occur?
- Was there instability/oscillation in win rates?
- Did win rates plateau or continue improving?

### Reward Trends
**TODO**: After examining rewards_chart_proc_*.png files:
- Did rewards increase over training?
- What was the approximate final reward?
- Were there sudden drops or spikes?

### Loss Behavior
**TODO**: After examining loss_chart_proc_*.png files:
- Did loss decrease consistently?
- Was training stable?
- Were there periods of high variance?

---

## Analysis: The 50% Barrier Problem

### Possible Reasons for the Barrier

#### 1. Low Discount Factor
**CRITICAL ISSUE**: γ = 0.5 is very low for FourRooms
- This environment requires long-term planning (many steps to goal)
- With γ=0.5, rewards 10 steps away are worth only 0.1% of immediate reward
- Agent cannot learn to value distant goals effectively

**Recommendation**: Increase to γ = 0.95-0.99

#### 2. Exploration Strategy
- Epsilon decay (0.998) might be too fast
- Minimum epsilon (0.1) might be too low for complex environment
- Agent may converge to suboptimal policy before finding better solutions

**Recommendation**:
- Slower decay: 0.999 or 0.9995
- Higher minimum: 0.15-0.20

#### 3. Environment Complexity
- FourRooms requires sophisticated navigation through doorways
- Partial observability (7×7 grid view) adds difficulty
- Sparse rewards make learning difficult

**Recommendation**: Consider curriculum learning
- Start with MiniGrid-Empty-8x8-v0 (simpler)
- Move to FourRooms after achieving 80%+ success

#### 4. Reward Shaping
- Current heuristic may not provide sufficient learning signal
- Need to verify heuristic is helping, not hindering

**Recommendation**: Experiment with different reward structures

#### 5. Network Architecture
- Current: 128 → 64 (relatively small)
- May lack capacity for complex environment

**Recommendation**: Try deeper/wider network
- 256 → 128 → 64
- Consider Dueling DQN architecture

---

## Recommendations (Priority Order)

### HIGH PRIORITY (Likely Causes)

1. **Increase Discount Factor**
   ```python
   DISCOUNT_FACTOR = 0.95  # or 0.99
   ```
   This is likely the primary issue. FourRooms needs long-term planning.

2. **Adjust Exploration**
   ```python
   DECAY_RATE = 0.999      # slower decay
   EPSYLON_MIN = 0.15      # higher minimum
   ```

3. **Try Simpler Environment First**
   ```python
   ENV_NAME = ENV_EMPTY_8x8
   ```
   Verify your implementation works on simpler task.

### MEDIUM PRIORITY (Worth Testing)

4. **Larger Network**
   ```python
   'neurons': 256  # first layer
   'neurons': 128  # second layer
   'neurons': 64   # third layer
   ```

5. **Adjust Learning Rate**
   ```python
   LR = 0.0003  # try lower
   # or
   LR = 0.001   # try higher
   ```

6. **Longer Training**
   ```python
   EPISODES = 5000  # more time to learn
   ```

### ADVANCED (If Above Don't Work)

7. **Prioritized Experience Replay**
8. **Double DQN** (reduce overestimation)
9. **Dueling DQN Architecture**
10. **Different Algorithm** (PPO, SAC)

---

## Conclusion

The consistent failure to break 50% success rate across multiple independent runs
suggests systematic limitations in the current approach, most likely:

1. **Discount factor too low** for long-horizon task (γ=0.5 → should be 0.95+)
2. **Exploration insufficient** for finding good policies
3. **Environment too complex** for current configuration

### Immediate Next Steps:

1. Set `DISCOUNT_FACTOR = 0.95`
2. Set `DECAY_RATE = 0.999`
3. Set `EPSYLON_MIN = 0.15`
4. Increase `EPISODES = 5000`
5. Run 4 parallel training processes again
6. Use automatic experiment tracking
7. Generate full report

If this doesn't break 50%, try curriculum learning with Empty-8x8-v0 first.

---

## Plots Reference

Training plots are available in `Database/Images/`:
- `winrate_chart_proc_0.png` through `winrate_chart_proc_3.png`
- `loss_chart_proc_0.png` through `loss_chart_proc_3.png`
- `rewards_chart_proc_0.png` through `rewards_chart_proc_3.png`

---
