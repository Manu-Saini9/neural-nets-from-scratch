"""
Author: Manu Saini (7092158) and Kev Akpinar (7159320)
Course: Machine Learning Project 4
Task: CNN for CelebA Smiling Classification

Description:
This file implements a simple Convolutional Neural Network (CNN)
from scratch using NumPy for binary image classification on the
CelebA smiling dataset.

Architecture:
Input -> Conv -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> Output -> Sigmoid
"""

import os
import json
import numpy as np


class SimpleCNN:
    """
    A simple CNN for binary classification.

    Network structure:
    - Convolution layer
    - ReLU activation
    - Max-pooling layer
    - Fully connected hidden layer
    - ReLU activation
    - Output layer
    - Sigmoid activation
    """

    def __init__(
        self,
        input_shape=(32, 32, 3),
        num_filters=8,
        filter_size=3,
        hidden_dim=64,
        learning_rate=0.001,
        seed=42,
    ):
        """
        Initialize the CNN parameters and hyperparameters.

        Parameters:
        - input_shape: Shape of the input image (height, width, channels)
        - num_filters: Number of convolution filters
        - filter_size: Height/width of each square filter
        - hidden_dim: Number of neurons in the fully connected hidden layer
        - learning_rate: Gradient descent step size
        - seed: Random seed for reproducibility
        """
        # Store model settings
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Random generator for reproducible initialization/shuffling
        self.rng = np.random.default_rng(seed)

        # Extract input dimensions
        h, w, c = input_shape

        # Initialize convolution filters
        # Shape: (number of filters, filter height, filter width, channels)
        self.filters = self.rng.normal(
            0.0, 0.1, size=(num_filters, filter_size, filter_size, c)
        ).astype(np.float32)

        # Initialize one bias value for each convolution filter
        self.conv_bias = np.zeros((num_filters,), dtype=np.float32)

        # Compute output size after convolution
        conv_h = h - filter_size + 1
        conv_w = w - filter_size + 1

        # Compute output size after 2x2 max-pooling with stride 2
        pool_h = conv_h // 2
        pool_w = conv_w // 2

        # Total number of values after flattening pooled feature maps
        flattened_dim = pool_h * pool_w * num_filters

        # Initialize first fully connected layer weights using He-style scaling
        self.W1 = self.rng.normal(
            0.0, np.sqrt(2.0 / flattened_dim), size=(flattened_dim, hidden_dim)
        ).astype(np.float32)

        # Bias for hidden fully connected layer
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        # Initialize output layer weights
        self.W2 = self.rng.normal(
            0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1)
        ).astype(np.float32)

        # Bias for output layer
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        # Store best model parameters based on validation accuracy
        self.best_params = None
        self.best_val_accuracy = 0.0

    def relu(self, x):
        """Apply ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Compute derivative of ReLU for backpropagation"""
        return (x > 0).astype(np.float32)

    def sigmoid(self, x):
        """
        Apply sigmoid activation.

        Clipping is used to prevent overflow in exp() for large values.
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.

        This is used for binary classification and compares
        predicted probabilities against true labels.
        """
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(
            -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        )

    def conv_forward_single(self, x):
        """
        Perform convolution on a single input image.

        Parameters:
        - x: Input image of shape (H, W, C)

        Returns:
        - Output feature maps of shape (num_filters, out_h, out_w)
        """
        H, W, C = x.shape
        F = self.filter_size
        out_h = H - F + 1
        out_w = W - F + 1

        # Output tensor for all filters
        out = np.zeros((self.num_filters, out_h, out_w), dtype=np.float32)

        # Apply each filter to the image
        for f in range(self.num_filters):
            filt = self.filters[f]
            bias = self.conv_bias[f]

            # Slide the filter spatially across the image
            for i in range(out_h):
                for j in range(out_w):
                    region = x[i:i + F, j:j + F, :]
                    out[f, i, j] = np.sum(region * filt) + bias

        return out

    def maxpool_forward(self, x):
        """
        Perform 2x2 max-pooling with stride 2.

        Parameters:
        - x: Input feature maps of shape (num_filters, H, W)

        Returns:
        - pooled: Downsampled feature maps
        - max_indices: Indices of max elements used later for backpropagation
        """
        num_filters, H, W = x.shape
        out_h = H // 2
        out_w = W // 2

        # Output after pooling
        pooled = np.zeros((num_filters, out_h, out_w), dtype=np.float32)

        # Stores where each max value came from
        max_indices = {}

        for f in range(num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * 2
                    h_end = h_start + 2
                    w_start = j * 2
                    w_end = w_start + 2

                    # Current 2x2 pooling region
                    region = x[f, h_start:h_end, w_start:w_end]

                    # Find location of the maximum value inside this region
                    max_pos = np.unravel_index(np.argmax(region), region.shape)

                    # Save max value to pooled output
                    pooled[f, i, j] = region[max_pos]

                    # Save original location of max value for backpropagation
                    max_indices[(f, i, j)] = (h_start + max_pos[0], w_start + max_pos[1])

        return pooled, max_indices

    def forward_single(self, x):
        """
        Perform a full forward pass for one input image.

        Steps:
        1. Convolution
        2. ReLU
        3. Max-pooling
        4. Flatten
        5. Fully connected hidden layer
        6. ReLU
        7. Output layer
        8. Sigmoid

        Returns:
        - Final predicted probability
        - Cache of intermediate values for backpropagation
        """
        # Convolution layer
        conv = self.conv_forward_single(x)

        # Activation after convolution
        relu_conv = self.relu(conv)

        # Max-pooling layer
        pooled, max_indices = self.maxpool_forward(relu_conv)

        # Flatten pooled feature maps into a row vector
        flat = pooled.reshape(1, -1)

        # Hidden fully connected layer
        z1 = np.dot(flat, self.W1) + self.b1
        a1 = self.relu(z1)

        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        # Save all important intermediate results for backward pass
        cache = {
            "x": x,
            "conv": conv,
            "relu_conv": relu_conv,
            "pooled": pooled,
            "max_indices": max_indices,
            "flat": flat,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
        }
        return a2, cache

    def backward_single(self, y_true, cache):
        """
        Perform backpropagation for a single training example.

        Parameters:
        - y_true: True label
        - cache: Intermediate results saved during forward pass

        Returns:
        - Dictionary of gradients for all trainable parameters
        """
        # Unpack cached forward-pass values
        x = cache["x"]
        conv = cache["conv"]
        relu_conv = cache["relu_conv"]
        pooled = cache["pooled"]
        max_indices = cache["max_indices"]
        flat = cache["flat"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        a2 = cache["a2"]

        # Output Layer Backpropagation 
        # Gradient of loss with respect to output layer pre-activation
        dz2 = a2 - y_true

        # Gradients for output layer weights and bias
        dW2 = np.dot(a1.T, dz2)
        db2 = dz2

        # Backpropagate into hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(z1)

        # Gradients for first fully connected layer
        dW1 = np.dot(flat.T, dz1)
        db1 = dz1

        # Backpropagate into flattened pooled representation
        dflat = np.dot(dz1, self.W1.T)
        dpooled = dflat.reshape(pooled.shape)

        # Max-Pooling Backpropagation 
        # Only the positions that were maximum during forward pass
        # receive the gradient
        drelu_conv = np.zeros_like(relu_conv, dtype=np.float32)
        for (f, i, j), (mi, mj) in max_indices.items():
            drelu_conv[f, mi, mj] = dpooled[f, i, j]

        #  ReLU Backpropagation 
        # Pass gradients only where convolution output was positive
        dconv = drelu_conv * self.relu_derivative(conv)

        # Convolution Layer Backpropagation 
        # Gradients for filters and convolution biases
        dfilters = np.zeros_like(self.filters, dtype=np.float32)
        dconv_bias = np.zeros_like(self.conv_bias, dtype=np.float32)

        F = self.filter_size
        out_h = dconv.shape[1]
        out_w = dconv.shape[2]

        for f in range(self.num_filters):
            dconv_bias[f] = np.sum(dconv[f])
            for i in range(out_h):
                for j in range(out_w):
                    region = x[i:i + F, j:j + F, :]
                    dfilters[f] += dconv[f, i, j] * region

        # Return all computed gradients
        grads = {
            "dW2": dW2,
            "db2": db2,
            "dW1": dW1,
            "db1": db1,
            "dfilters": dfilters,
            "dconv_bias": dconv_bias,
        }
        return grads

    def update_params(self, grads):
        """
        Update all trainable parameters using gradient descent.
        """
        self.W2 -= self.learning_rate * grads["dW2"]
        self.b2 -= self.learning_rate * grads["db2"]
        self.W1 -= self.learning_rate * grads["dW1"]
        self.b1 -= self.learning_rate * grads["db1"]
        self.filters -= self.learning_rate * grads["dfilters"]
        self.conv_bias -= self.learning_rate * grads["dconv_bias"]

    def predict_proba(self, X):
        """
        Predict output probabilities for a batch of input images.
        """
        probs = []
        for i in range(X.shape[0]):
            prob, _ = self.forward_single(X[i])
            probs.append(prob[0, 0])
        return np.array(probs, dtype=np.float32).reshape(-1, 1)

    def predict(self, X):
        """
        Convert predicted probabilities into binary class labels
        using threshold 0.5.
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.float32)

    def compute_accuracy(self, X, y):
        """
        Compute classification accuracy as a percentage.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y) * 100.0)

    def save_best_params(self):
        """
        Save the current model parameters as the best parameters.
        Used when validation accuracy improves.
        """
        self.best_params = {
            "filters": self.filters.copy(),
            "conv_bias": self.conv_bias.copy(),
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def restore_best_params(self):
        """
        Restore the best saved model parameters after training.
        """
        if self.best_params is not None:
            self.filters = self.best_params["filters"].copy()
            self.conv_bias = self.best_params["conv_bias"].copy()
            self.W1 = self.best_params["W1"].copy()
            self.b1 = self.best_params["b1"].copy()
            self.W2 = self.best_params["W2"].copy()
            self.b2 = self.best_params["b2"].copy()

    def train(self, X_train, y_train, X_val, y_val, epochs=5):
        """
        Train the CNN one sample at a time.

        Training procedure:
        - Shuffle training data each epoch
        - Perform forward pass
        - Compute loss
        - Perform backward pass
        - Update parameters
        - Evaluate full training and validation performance each epoch
        - Save best model based on validation accuracy
        """
        # Store training history over epochs
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            total_loss = 0.0

            # Train on each sample one by one
            for i in range(n_samples):
                x_i = X_train_shuffled[i]
                y_i = y_train_shuffled[i].reshape(1, 1)

                # Forward pass
                y_pred, cache = self.forward_single(x_i)

                # Compute loss
                loss = self.binary_cross_entropy(y_i, y_pred)
                total_loss += loss

                # Backward pass and parameter update
                grads = self.backward_single(y_i, cache)
                self.update_params(grads)

                # Progress update every 500 samples
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{n_samples} training samples")

            # Evaluate performance on full training and validation sets
            train_probs = self.predict_proba(X_train)
            val_probs = self.predict_proba(X_val)

            train_loss = self.binary_cross_entropy(y_train, train_probs)
            val_loss = self.binary_cross_entropy(y_val, val_probs)

            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)

            # Save metrics
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # Save best model if validation accuracy improves
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.save_best_params()

            # Print epoch summary
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
            )

        # Restore the best model parameters after training ends
        self.restore_best_params()
        print(f"\nBest validation accuracy: {self.best_val_accuracy:.2f}%")
        return history


def load_processed_image_data(cache_dir):
    """
    Load preprocessed image and label arrays from disk.

    Expected files:
    - X_train_img.npy
    - X_val_img.npy
    - X_test_img.npy
    - y_train.npy
    - y_val.npy
    - y_test.npy
    """
    X_train = np.load(os.path.join(cache_dir, "X_train_img.npy"))
    X_val = np.load(os.path.join(cache_dir, "X_val_img.npy"))
    X_test = np.load(os.path.join(cache_dir, "X_test_img.npy"))

    y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
    y_val = np.load(os.path.join(cache_dir, "y_val.npy"))
    y_test = np.load(os.path.join(cache_dir, "y_test.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Get directory paths for loading data and saving results
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(BASE_DIR, "processed_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    print("Loading processed image dataset from:", CACHE_DIR)

    # Load training, validation, and test datasets
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_image_data(CACHE_DIR)

    # Print data shapes for confirmation
    print("X_train shape:", X_train.shape)
    print("X_val shape  :", X_val.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)

    # Create CNN model
    cnn = SimpleCNN(
        input_shape=(32, 32, 3),
        num_filters=16,
        filter_size=3,
        hidden_dim=64,
        learning_rate=0.001,
        seed=42,
    )

    # Train the model and store learning history
    history = cnn.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=15,
    )

    # Evaluate final model on the test set
    test_probs = cnn.predict_proba(X_test)
    test_loss = cnn.binary_cross_entropy(y_test, test_probs)
    test_acc = cnn.compute_accuracy(X_test, y_test)

    print("\nFinal Test Loss: {:.4f}".format(test_loss))
    print("Final Test Accuracy: {:.2f}%".format(test_acc))

    # Create results directory if needed
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate final binary predictions
    y_test_pred = cnn.predict(X_test)

    # Save test labels and predictions
    np.save(os.path.join(RESULTS_DIR, "cnn_y_true.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, "cnn_y_pred.npy"), y_test_pred)

    # Save training history
    with open(os.path.join(RESULTS_DIR, "cnn_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save final model metrics
    with open(os.path.join(RESULTS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump({
            "model": "CNN",
            "num_filters": 16,
            "filter_size": 3,
            "hidden_dim": 64,
            "learning_rate": 0.001,
            "epochs": 15,
            "best_validation_accuracy": cnn.best_val_accuracy,
            "final_test_loss": test_loss,
            "final_test_accuracy": test_acc
        }, f, indent=4)

    print("CNN results saved to:", RESULTS_DIR)