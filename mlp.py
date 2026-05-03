"""
Author: Manu Saini (7092158) and Kev Akpinar (7159320)
Course: Machine Learning Project 4
Task: MLP for CelebA Smiling Classification

Description:
This file implements a 1-hidden-layer Multilayer Perceptron (MLP)
from scratch using NumPy for binary classification on the
CelebA smiling dataset. The network uses ReLU activation in the
hidden layer, sigmoid activation in the output layer, He-based
weight initialization, and validation-based checkpointing to
retain the best-performing model.
"""

import os
import json
import numpy as np


class MLP:
    """
    A 1-hidden-layer Multilayer Perceptron (MLP) for binary classification.

    Architecture:
    Input → Fully Connected → ReLU → Fully Connected → Sigmoid → Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        seed: int = 42,
    ) -> None:
        """
        Initialize the model hyperparameters and trainable parameters.

        Parameters:
        - input_dim: Number of input features
        - hidden_dim: Number of neurons in the hidden layer
        - learning_rate: Step size used during gradient descent
        - seed: Random seed for reproducible initialization
        """
        # Store configuration values
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Random generator for reproducible weight initialization and shuffling
        self.rng = np.random.default_rng(seed)

        # Initialize hidden layer weights using He initialization,
        # which works well with ReLU activation
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        # Initialize output layer weights with a smaller variance
        self.W2 = self.rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        # Keep track of the best model parameters based on validation accuracy
        self.best_W1 = self.W1.copy()
        self.best_b1 = self.b1.copy()
        self.best_W2 = self.W2.copy()
        self.best_b2 = self.b2.copy()
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

        Values are clipped first to avoid overflow in exp().
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray):
        """
        Perform a forward pass through the network.

        Steps:
        1. Compute hidden layer linear transformation
        2. Apply ReLU activation
        3. Compute output layer linear transformation
        4. Apply sigmoid activation

        Returns:
        - z1: Hidden layer pre-activation values
        - a1: Hidden layer activations
        - z2: Output layer pre-activation values
        - a2: Output probabilities
        """
        # Hidden layer computation
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)

        # Output layer computation
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        return z1, a1, z2, a2

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.

        This measures how close predicted probabilities are
        to the true binary labels.
        """
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for the input samples.
        """
        _, _, _, a2 = self.forward(X)
        return a2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Convert predicted probabilities into binary labels.

        Threshold:
        - probability >= 0.5 -> class 1
        - probability < 0.5  -> class 0
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.float32)

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy as a percentage.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y) * 100.0)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 40,
        batch_size: int = 64,
    ) -> dict:
        """
        Train the MLP using mini-batch gradient descent.

        Training workflow:
        - Shuffle the training data each epoch
        - Process data in mini-batches
        - Perform forward pass
        - Compute gradients with backpropagation
        - Update model parameters
        - Evaluate on training and validation sets
        - Save the best model based on validation accuracy
        """
        # Store learning history across all epochs
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # Number of training samples
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Process training data in mini-batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward pass for the current mini-batch
                z1, a1, z2, a2 = self.forward(X_batch)

                # Current batch size
                m = X_batch.shape[0]

                # output layer gradients 
                dz2 = a2 - y_batch
                dW2 = np.dot(a1.T, dz2) / m
                db2 = np.sum(dz2, axis=0, keepdims=True) / m

                #  Hidden layer gradients 
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.relu_derivative(z1)
                dW1 = np.dot(X_batch.T, dz1) / m
                db1 = np.sum(dz1, axis=0, keepdims=True) / m

                # Parameter update 
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1

            # Compute full training and validation predictions after each epoch
            train_probs = self.predict_proba(X_train)
            val_probs = self.predict_proba(X_val)

            # Compute loss values
            train_loss = self.binary_cross_entropy(y_train, train_probs)
            val_loss = self.binary_cross_entropy(y_val, val_probs)

            # Compute accuracy values
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)

            # Save metrics into history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # Save best parameters when validation accuracy improves
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_W1 = self.W1.copy()
                self.best_b1 = self.b1.copy()
                self.best_W2 = self.W2.copy()
                self.best_b2 = self.b2.copy()

            # Display epoch summary
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
            )

        # Restore the best model found during validation
        self.W1 = self.best_W1.copy()
        self.b1 = self.best_b1.copy()
        self.W2 = self.best_W2.copy()
        self.b2 = self.best_b2.copy()

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
    # Get file system paths for data loading and result saving
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(BASE_DIR, "processed_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    print("Loading processed dataset from:", CACHE_DIR)

    # Load processed datasets
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(CACHE_DIR)

    # Print dataset shapes for verification
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape  :", X_val.shape)
    print("y_val shape  :", y_val.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)

    # Create the MLP model
    mlp = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        learning_rate=0.001,
        seed=42,
    )

    # Train the model and collect training history
    history = mlp.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=40,
        batch_size=64,
    )

    # Evaluate the trained model on the test set
    test_accuracy = mlp.compute_accuracy(X_test, y_test)
    test_probs = mlp.predict_proba(X_test)
    test_loss = mlp.binary_cross_entropy(y_test, test_probs)

    print("\nFinal Test Loss: {:.4f}".format(test_loss))
    print("Final Test Accuracy: {:.2f}%".format(test_accuracy))

    # Create results directory if needed
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate final test predictions
    y_test_pred = mlp.predict(X_test)

    # Save true labels and predicted labels
    np.save(os.path.join(RESULTS_DIR, "mlp_y_true.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, "mlp_y_pred.npy"), y_test_pred)

    # Save training history
    with open(os.path.join(RESULTS_DIR, "mlp_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save final metrics
    with open(os.path.join(RESULTS_DIR, "mlp_metrics.json"), "w") as f:
        json.dump({
            "model": "MLP",
            "hidden_dim": 256,
            "learning_rate": 0.001,
            "epochs": 40,
            "batch_size": 64,
            "best_validation_accuracy": mlp.best_val_accuracy,
            "final_test_loss": test_loss,
            "final_test_accuracy": test_accuracy
        }, f, indent=4)

    print("MLP results saved to:", RESULTS_DIR)