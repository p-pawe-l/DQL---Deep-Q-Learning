import numpy as np


def preprocess_minigrid(obs):
        image = obs['image']
        flat = image.flatten().astype(np.float32)
        flat /= 10.0
        return np.expand_dims(flat, axis=0)

def preprocess_LunarLander(obs):
        if isinstance(obs, tuple):
                obs = obs[0]
        
        return np.expand_dims(obs, axis=0)