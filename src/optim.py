from dataclasses import dataclass

@dataclass
class GradientDescentOptimizer:
    lr: float # Learning rate

    def step(self, model, dL_dw: float, dL_db: float):
        """
        Update model parameters in-place
            w := w - lr * dL_dw
            b := b - lr * dL_db
        """
        model.w -= self.lr * dL_dw
        model.b -= self.lr * dL_db
