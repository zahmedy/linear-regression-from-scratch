Overview
--------
A concise architecture and step-by-step plan to build linear regression from scratch (no ML frameworks). This README explains the components, project layout, and the minimal code/workflow needed to train and evaluate a simple linear model y = wx + b using mean squared error and gradient descent.

Architecture
------------
- Data: generate or load a 1D dataset (x, y) and split into train/validation.
- Model: simple parametric model with parameters w (weight) and b (bias).
- Loss: mean squared error (MSE) between predictions and targets.
- Optimizer: vanilla gradient descent (compute gradients analytically).
- Training loop: forward pass -> compute loss -> backward pass -> parameter update.
- Evaluation: compute loss on validation set and plot predictions vs. targets.

Project layout
--------------
- data/
    - generate.py         # utilities to synthesize or load data
- src/
    - model.py            # Model class: stores parameters, predict method
    - loss.py             # MSE implementation and gradient computation
    - optim.py            # Gradient descent optimizer (update step)
    - train.py            # Training loop and checkpointing
    - eval.py             # Evaluation and plotting
- notebooks/
    - demo.ipynb          # Interactive tutorial and visualizations
- README.md

Step-by-step implementation plan
--------------------------------
1. Data
     - Create a function to generate linear data with optional noise.
     - Normalize or standardize x if desired.
     - Split into train/validation sets.

2. Model
     - Represent parameters as simple floats or NumPy arrays: w, b.
     - Implement predict(x): return w*x + b.

3. Loss
     - Implement MSE: L = mean((y_pred - y_true)^2).
     - Compute analytical gradients:
         - dL/dw = (2/N) * sum((y_pred - y_true) * x)
         - dL/db = (2/N) * sum(y_pred - y_true)

4. Optimizer
     - Implement gradient descent: w -= lr * dL/dw ; b -= lr * dL/db
     - Optionally add momentum or learning rate scheduling.

5. Training loop
     - For each epoch:
         - Forward: compute y_pred
         - Loss: compute MSE
         - Backward: compute gradients using loss formulas
         - Update: apply optimizer to parameters
         - Log training and validation loss periodically

6. Evaluation & visualization
     - Plot predicted line and scatter of true points.
     - Report final training/validation MSE and parameter values.
     - Optionally compute R^2 score.

Example usage
-------------
- Generate data:
    - python src/data/generate.py
- Train model:
    - python src/train.py
- Evaluate:
    - python src/eval.py

Tips & extensions
-----------------
- Add mini-batch gradient descent instead of full-batch.
- Implement L2 regularization (weight decay).
- Extend to multivariate linear regression (matrix form).
- Compare your implementation against scikit-learn's LinearRegression.
- Add unit tests for gradients and loss computation.

Expected learning outcomes
--------------------------
- Understand the full training loop from data to prediction.
- Derive and implement gradients analytically.
- Gain intuition for learning rate, convergence, and overfitting.
- Be able to extend to more complex models and optimizers.
