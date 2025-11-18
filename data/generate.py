import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

def generate_linear_data(
    n_samples: int = 100,
    w_true: float = 2.0,
    b_true: float = 1.0,
    noise_std: float = 0.1,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """
    Generate 1D linear data: y = w_true * x + b_true + noise
    and split into train/validation.
    """
    rng = np.random.default_rng(seed)

    # 1) Create random x values (e.g., between -1 and 1)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 1))

    # 2) Compute noiseless y = w_true * x + b_true
    y_clean = w_true * X + b_true

    # 3) Add Gaussian noise
    noise = rng.normal(0.0, noise_std, size=y_clean.shape)
    y = y_clean + noise

    # 4) Split into train / val
    n_val = int(n_samples * val_ratio)
    # simple split: last n_val as val
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    return Dataset(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
