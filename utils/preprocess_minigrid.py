import numpy as np


def preprocess(obs):
        image = obs['image']
        flat = image.flatten().astype(np.float32)
        flat /= 10.0
        return np.expand_dims(flat, axis=0)