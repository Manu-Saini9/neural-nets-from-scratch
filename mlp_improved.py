"""
Author: Manu Saini (7092158) and Kev Akpinar (7159320)
Course: Machine Learning Project
Task: Improved 2-Hidden-Layer MLP for CelebA Smiling Classification

Description:
This file implements an improved Multilayer Perceptron (MLP)
from scratch using NumPy for binary classification on the
CelebA smiling dataset. The model uses two hidden layers,
ReLU activation, dropout regularization, He initialization,
and validation-based checkpointing to preserve the best model.
"""

import os
import json
import numpy as np


class MLP:
    """
    An improved 2-hidden-layer Multilayer Perceptron (MLP)
    for binary classification.

    Architecture:
    Input → Fully Connected → ReLU → Dropout →
    Fully Connected → ReLU → Dropout →
    Fully Connected → Sigmoid → Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        seed: int = 42,
    ) -> None:
        """
        Initialize model hyperparameters and trainable parameters.

        Parameters:
        - input_dim: Number of input features
        - hidden_dim1: Number of neurons in the first hidden layer
        - hidden_dim2: Number of neurons in the second hidden layer
        - learning_rate: Gradient descent step size
        - dropout_rate: Fraction of neurons dropped during training
        - seed: Random seed for reproducibility
        """
        # Store model configuration
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Random number generator for reproducible initialization and shuffling
        self.rng = np.random.default_rng(seed)

        # ----- Parameter initialization -----
        # He initialization is used because the hidden layers use ReLU activation
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim1))
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)

        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim1), size=(hidden_dim1, hidden_dim2))
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)

        # Output layer uses a smaller variance before sigmoid activation
        self.W3 = self.rng.normal(0.0, np.sqrt(1.0 / hidden_dim2), size=(hidden_dim2, 1))
        self.b3 = np.zeros((1, 1), dtype=np.float32)

        # Save the best-performing parameters based on validation accuracy
        self.best_params = None
        self.best_val_accuracy = 0.0

    def relu(self, z: np.ndarray) -> np.ndarray:
        """
        Apply ReLU activation element-wise.

        ReLU(z) = max(0, z)
        """
        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of ReLU for backpropagation.
        """
        return (z > 0).astype(np.float32)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation to convert logits into probabilities.

        Values are clipped first to avoid numerical overflow.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def apply_dropout(self, a: np.ndarray, training: bool):
        """
        Apply inverted dropout to an activation matrix during training.

        Parameters:
        - a: Activation values
        - training: Whether the model is currently in training mode

        Returns:
        - a_dropped: Activations after dropout
        - mask: Dropout mask used during training
        """
        # Do not apply dropout during evaluation or if dropout is disabled
        if not training or self.dropout_rate <= 0.0:
            mask = np.ones_like(a)
            return a, mask

        # Keep probability for each neuron
        keep_prob = 1.0 - self.dropout_rate

        # Randomly generate dropout mask
        mask = (self.rng.random(a.shape) < keep_prob).astype(np.float32)

        # Inverted dropout scaling keeps expected activation magnitude unchanged
        a_dropped = (a * mask) / keep_prob
        return a_dropped, mask

    def forward(self, X: np.ndarray, training: bool = False):
        """
        Perform a forward pass through the network.

        Steps:
        1. First hidden layer linear transformation
        2. ReLU activation
        3. Dropout
        4. Second hidden layer linear transformation
        5. ReLU activation
        6. Dropout
        7. Output layer linear transformation
        8. Sigmoid activation

        Returns:
        - cache: Tuple containing intermediate values needed for backpropagation
        """
        # ----- First hidden layer -----
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        a1, dmask1 = self.apply_dropout(a1, training)

        # ----- Second hidden layer -----
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        a2, dmask2 = self.apply_dropout(a2, training)

        # ----- Output layer -----
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.sigmoid(z3)

        # Cache intermediate values for gradient computation
        cache = (z1, a1, dmask1, z2, a2, dmask2, z3, a3)
        return cache

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        """
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output probabilities for the input samples.
        """
        _, _, _, _, _, _, _, a3 = self.forward(X, training=False)
        return a3

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Convert predicted probabilities into binary labels
        using a threshold of 0.5.
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.float32)

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy as a percentage.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y) * 100.0)

    def save_best_params(self):
        """
        Save the current model parameters as the best parameters.
        This is called when validation accuracy improves.
        """
        self.best_params = {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
        }

    def restore_best_params(self):
        """
        Restore the best saved model parameters after training.
        """
        if self.best_params is not None:
            self.W1 = self.best_params["W1"].copy()
            self.b1 = self.best_params["b1"].copy()
            self.W2 = self.best_params["W2"].copy()
            self.b2 = self.best_params["b2"].copy()
            self.W3 = self.best_params["W3"].copy()
            self.b3 = self.best_params["b3"].copy()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> dict:
        """
        Train the improved MLP using mini-batch gradient descent.

        Training workflow:
        - Shuffle data at the beginning of each epoch
        - Process mini-batches
        - Perform forward pass with dropout
        - Compute gradients with backpropagation
        - Update parameters
        - Evaluate on training and validation sets
        - Save the best model based on validation accuracy
        """
        # Store training progress for later plotting/analysis
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training set each epoch
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Process data in mini-batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward pass with dropout enabled
                z1, a1, dmask1, z2, a2, dmask2, z3, a3 = self.forward(X_batch, training=True)

                # Mini-batch size
                m = X_batch.shape[0]

                #  Output layer gradients 
                dz3 = a3 - y_batch
                dW3 = np.dot(a2.T, dz3) / m
                db3 = np.sum(dz3, axis=0, keepdims=True) / m

                # Second hidden layer gradients 
                da2 = np.dot(dz3, self.W3.T)
                da2 = da2 * dmask2 / (1.0 - self.dropout_rate)
                dz2 = da2 * self.relu_derivative(z2)
                dW2 = np.dot(a1.T, dz2) / m
                db2 = np.sum(dz2, axis=0, keepdims=True) / m

                # First hidden layer gradients 
                da1 = np.dot(dz2, self.W2.T)
                da1 = da1 * dmask1 / (1.0 - self.dropout_rate)
                dz1 = da1 * self.relu_derivative(z1)
                dW1 = np.dot(X_batch.T, dz1) / m
                db1 = np.sum(dz1, axis=0, keepdims=True) / m

                # Gradient descent parameter updates 
                self.W3 -= self.learning_rate * dW3
                self.b3 -= self.learning_rate * db3
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1

            # Evaluate full training and validation sets after each epoch
            train_probs = self.predict_proba(X_train)
            val_probs = self.predict_proba(X_val)

            train_loss = self.binary_cross_entropy(y_train, train_probs)
            val_loss = self.binary_cross_entropy(y_val, val_probs)

            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)

            # Save epoch metrics
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # Save model parameters if validation accuracy improves
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.save_best_params()

            # Print epoch summary
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
            )

        # Restore best validation model after training completes
        self.restore_best_params()
        print(f"\nBest validation accuracy: {self.best_val_accuracy:.2f}%")
        return history


def load_processed_data(cache_dir: str):
    """
    Load preprocessed train, validation, and test datasets from disk.

    Returns:
    - X_train, X_val, X_test: Input feature matrices
    - y_train, y_val, y_test: Corresponding label arrays
    """
    X_train = np.load(os.path.join(cache_dir, "X_train_flat.npy"))
    X_val = np.load(os.path.join(cache_dir, "X_val_flat.npy"))
    X_test = np.load(os.path.join(cache_dir, "X_test_flat.npy"))

    y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
    y_val = np.load(os.path.join(cache_dir, "y_val.npy"))
    y_test = np.load(os.path.join(cache_dir, "y_test.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Define file system paths for dataset loading and result storage
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(BASE_DIR, "processed_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    print("Loading processed dataset from:", CACHE_DIR)

    # Load preprocessed train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(CACHE_DIR)

    # Create improved MLP model
    mlp = MLP(
        input_dim=X_train.shape[1],
        hidden_dim1=256,
        hidden_dim2=128,
        learning_rate=0.0005,
        dropout_rate=0.2,
        seed=42,
    )

    # Train the model and store training history
    history = mlp.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=64,
    )

    # Evaluate final model on the test set
    test_probs = mlp.predict_proba(X_test)
    test_loss = mlp.binary_cross_entropy(y_test, test_probs)
    test_acc = mlp.compute_accuracy(X_test, y_test)

    print("\nFinal Test Loss: {:.4f}".format(test_loss))
    print("Final Test Accuracy: {:.2f}%".format(test_acc))

    # Create results directory if not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate final binary predictions
    y_test_pred = mlp.predict(X_test)

    # Save true labels and predicted labels
    np.save(os.path.join(RESULTS_DIR, "mlp_improved_y_true.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, "mlp_improved_y_pred.npy"), y_test_pred)

    # Save training history
    with open(os.path.join(RESULTS_DIR, "mlp_improved_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save final model metrics
    with open(os.path.join(RESULTS_DIR, "mlp_improved_metrics.json"), "w") as f:
        json.dump({
            "model": "Improved MLP",
            "hidden_dim1": 256,
            "hidden_dim2": 128,
            "learning_rate": 0.0005,
            "dropout_rate": 0.2,
            "epochs": 50,
            "batch_size": 64,
            "best_validation_accuracy": mlp.best_val_accuracy,
            "final_test_loss": test_loss,
            "final_test_accuracy": test_acc
        }, f, indent=4)

    print("Improved MLP results saved to:", RESULTS_DIR)