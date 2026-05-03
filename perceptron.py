"""
Author: Manu Saini (7092158) and Kev Akpinar (7159320)
Course: Machine Learning Project 4
Task: Perceptron for CelebA Smiling Classification

Description:
This program trains a binary perceptron model from scratch to classify
the CelebA "Smiling" attribute using flattened image vectors. It loads
preprocessed training, validation, and test data, trains the perceptron,
tracks accuracy over epochs, restores the best model based on validation
accuracy, and saves the final predictions and metrics for later comparison.
"""

import os
import json
import numpy as np


class Perceptron:
    """
    A simple binary perceptron classifier.

    The perceptron learns a linear decision boundary of the form:
        z = XW + b
    and applies a step activation function to produce binary predictions.
    """

    def __init__(self, input_dim: int, learning_rate: float = 0.0005, seed: int = 42) -> None:
        """
        Initialize perceptron parameters.

        Parameters:
        - input_dim: Number of input features.
        - learning_rate: Step size used during weight updates.
        - seed: Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Initialize weights with small random values and bias to zero.
        self.weights = self.rng.normal(loc=0.0, scale=0.01, size=(input_dim, 1))
        self.bias = 0.0

        # Store the best-performing parameters based on validation accuracy.
        self.best_weights = self.weights.copy()
        self.best_bias = self.bias
        self.best_val_accuracy = 0.0

    def step_function(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the binary step activation function.

        Returns 1 where z >= 0, otherwise returns 0.
        """
        return (z >= 0).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input samples.

        Parameters:
        - X: Input feature matrix of shape (n_samples, input_dim)

        Returns:
        - Binary predictions of shape (n_samples, 1)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.step_function(z)

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy as a percentage.

        Parameters:
        - X: Input features
        - y: True labels

        Returns:
        - Accuracy in percentage
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y) * 100.0)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
    ) -> dict:
        """
        Train the perceptron using the perceptron learning rule.

        During training:
        - The training data is shuffled each epoch.
        - Each sample is processed one at a time.
        - Weights and bias are updated only when a misclassification occurs.
        - The best model is saved according to validation accuracy.

        Parameters:
        - X_train: Training feature matrix
        - y_train: Training labels
        - X_val: Validation feature matrix
        - y_val: Validation labels
        - epochs: Number of full training passes

        Returns:
        - history: Dictionary containing training/validation accuracy
                   and number of mistakes per epoch
        """
        history = {
            "train_accuracy": [],
            "val_accuracy": [],
            "mistakes_per_epoch": [],
        }

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            mistakes = 0

            # Shuffle training samples at the start of each epoch
            # to reduce ordering bias during learning.
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Process one sample at a time
            for i in range(n_samples):
                x_i = X_train_shuffled[i].reshape(1, -1)
                y_i = y_train_shuffled[i, 0]

                # Forward pass: compute raw score and predicted label
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = int(self.step_function(z)[0, 0])

                # Perceptron error term
                error = y_i - y_pred

                # Update only if prediction is incorrect
                if error != 0:
                    mistakes += 1
                    self.weights += self.learning_rate * x_i.T * error
                    self.bias += self.learning_rate * error

            # Evaluate performance after the epoch
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)

            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["mistakes_per_epoch"].append(mistakes)

            # Save the current model if validation accuracy improves
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_weights = self.weights.copy()
                self.best_bias = self.bias

            # Print epoch summary
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Mistakes: {mistakes:4d} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}%"
            )

        # Restore the best model found during training
        self.weights = self.best_weights.copy()
        self.bias = self.best_bias

        print(f"\nBest validation accuracy: {self.best_val_accuracy:.2f}%")
        return history


def load_processed_data(cache_dir: str) -> tuple:
    """
    Load preprocessed flattened feature vectors and labels from disk.

    Parameters:
    - cache_dir: Directory containing saved NumPy arrays

    Returns:
    - Tuple containing:
      X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train = np.load(os.path.join(cache_dir, "X_train_flat.npy"))
    X_val = np.load(os.path.join(cache_dir, "X_val_flat.npy"))
    X_test = np.load(os.path.join(cache_dir, "X_test_flat.npy"))

    y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
    y_val = np.load(os.path.join(cache_dir, "y_val.npy"))
    y_test = np.load(os.path.join(cache_dir, "y_test.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Define project directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(BASE_DIR, "processed_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    print("Loading processed dataset from:", CACHE_DIR)

    # Load training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(CACHE_DIR)

    # Print dataset shapes for verification
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape  :", X_val.shape)
    print("y_val shape  :", y_val.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)

    # Create perceptron model
    perceptron = Perceptron(
        input_dim=X_train.shape[1],
        learning_rate=0.001,
        seed=42,
    )

    # Train the perceptron
    history = perceptron.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=20,
    )

    # Evaluate the best model on the test set
    test_accuracy = perceptron.compute_accuracy(X_test, y_test)
    print("\nFinal Test Accuracy (best model): {:.2f}%".format(test_accuracy))

    # Create results directory if not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate predictions for the test set
    y_test_pred = perceptron.predict(X_test)

    # Save true labels and predictions for later comparison/visualization
    np.save(os.path.join(RESULTS_DIR, "perceptron_y_true.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, "perceptron_y_pred.npy"), y_test_pred)

    # Save training history as JSON
    with open(os.path.join(RESULTS_DIR, "perceptron_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save model settings and final evaluation metrics as JSON
    with open(os.path.join(RESULTS_DIR, "perceptron_metrics.json"), "w") as f:
        json.dump({
            "model": "Perceptron",
            "learning_rate": 0.0005,
            "epochs": 20,
            "best_validation_accuracy": perceptron.best_val_accuracy,
            "final_test_accuracy": test_accuracy
        }, f, indent=4)

    print("Perceptron results saved to:", RESULTS_DIR)