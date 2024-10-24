import numpy as np
from .baselines import AdaFairClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score


def mask_to_idx(mask: np.ndarray):
    return np.where(mask)[0]


def idx_to_mask(indices: np.ndarray, size: int):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def dict_info(d):
    info = ''
    for k, v in d.items():
        info += f'{k}: {v}\n'
    return info


def seed_generator(master_seed):
    """
    Generator function to produce random seeds based on a master seed.

    Parameters:
        master_seed (int): The master seed used to initialize the random number generator.

    Yields:
        int: A random seed generated using the master_seed.
    """
    rng = np.random.default_rng(master_seed)
    while True:
        yield rng.integers(np.iinfo(np.uint32).max)
