import numpy as np

def MINI_GRID_HEURISTIC_FOURROOMS(reward, prev_pos, current_pos, curiosity_table, episode, step_count) -> float:
        doing_nothing = False
        if prev_pos is not None: doing_nothing = np.array_equal(prev_pos, current_pos)

        cx, cy = int(current_pos[0]), int(current_pos[1])
        visits = curiosity_table[cx, cy]

        total_reward = 0.0

        if reward > 0: return 200.0
        
        total_reward -= 0.1
        
        if doing_nothing: total_reward -= 2.0

        curiosity_decay = max(0.3, 1.0 - (episode / 20000))

        if visits < 5: curiosity_bonus = curiosity_decay * 5.0
        else: curiosity_bonus = curiosity_decay * (3.0 / np.sqrt(visits))

        if not doing_nothing:
                curiosity_table[cx, cy] += 1
                total_reward += curiosity_bonus

        if step_count > 500: total_reward -= 0.5

        return total_reward


def MINI_GRID_HEURESTIC_LAVA(reward, done) -> float:
        if reward > 0: return 150
        # He jumps into lava
        elif reward == 0 and done: return -20
        return -0.05
        
                        
def LUNAR_LANDER_HEURESTIC(reward) -> float:
        return reward / 10.0