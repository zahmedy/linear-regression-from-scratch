import numpy as np
from data.generate import generate_linear_data
from model import LinearModel
from loss import mse_loss, mse_gradients
from optim import GradientDescentOptimizer

def train(epochs=200, lr=1.0):
    # 1. Generate synthetic training data
    data = generate_linear_data(n_samples=100, noise_std=0.1)

    # 2. Initialize model with random parameters
    rng = np.random.default_rng(0)
    w_init = rng.normal()
    b_init =rng.normal()
    model = LinearModel(w=w_init,b=b_init)

    # 3. Set up optimizer
    optimizer = GradientDescentOptimizer(lr=lr)

    for epoch in epochs:
        # Forward
        y_pred = model.predict(data.X_train)

        # Loss
        train_loss = mse_loss(y_pred, data.y_train)

        # Backward
        dL_dw, dL_db = mse_gradients(data.X_train, y_pred, data.y_train)

        # Update optimizer
        optimizer.step(model, dL_dw, dL_db)

        # Validation loss
        val_pred = model.predict(data.X_val)
        val_loss = mse_loss(val_pred, data.y_val)

        # Logging every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    print("\nTraining finished.")
    print(f"Learned w: {model.w:.4f}, b: {model.b:.4f}")     

    return model

if __name__ == "__main__":
    train()