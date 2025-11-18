import numpy as np
import matplotlib.pyplot as plt

from src.model import LinearModel
from data.generate import generate_linear_data

def visualize(model, n_samples=100):
    # 1. Regenerate the same data
    data = generate_linear_data(n_samples=n_samples, noise_std=0.1)

    X = data.X_train
    y = data.y_train

    # 2. Predications from your learned model 
    y_pred = model.predict(X)

    # 3. Plot scatter of true data
    plt.scatter(X, y, label="Date points", alpha=0.6)

    # 4. Plot the learned line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)

    plt.plot(X_line, y_line, color='red', label="Learned Line", linewidth=2)

    plt.title("Linear Regression Fit")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()