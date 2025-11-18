import numpy as np
from dataclasses import dataclass

@dataclass
class LinearModel:
    w: float
    b: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions: y = w*x = b
        X is expected to have a value (N, 1)
        """
        return self.w * X + self.b
    