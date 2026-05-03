"""
Author: Manu Saini (7092158) and Kev Akpinar (7159320)
Course: Machine Learning Project 4
Task: Compare model results using plots and confusion matrices

Description:
This script compares multiple trained models (Perceptron, MLP, Improved MLP, CNN)
by:
- Computing classification metrics (accuracy, precision, recall, F1-score)
- Generating confusion matrices
- Plotting accuracy/loss curves
- Creating comparison bar charts
- Saving results in CSV and JSON formats
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    """
    Load a JSON file from disk.

    Parameters:
    - path: Path to JSON file

    Returns:
    - Parsed JSON content as Python dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def confusion_matrix_binary(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.

    Matrix format:
        [[TN, FP],
         [FN, TP]]

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels

    Returns:
    - 2x2 NumPy array representing confusion matrix
    """
    # Ensure integer format and flatten arrays
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    # Compute confusion matrix components
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives

    return np.array([[tn, fp],
                     [fn, tp]])


def classification_metrics(y_true, y_pred):
    """
    Compute classification performance metrics.

    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Includes safeguards against division by zero.

    Returns:
    - Dictionary containing all metrics and confusion matrix
    """
    cm = confusion_matrix_binary(y_true, y_pred)

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    # Compute metrics with safety guards
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(cm, title, save_path):
    """
    Plot and save confusion matrix as an image.

    Parameters:
    - cm: Confusion matrix (2x2 array)
    - title: Title of the plot
    - save_path: File path to save the plot
    """
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    # Label axes
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    # Annotate each cell with its value
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # Save and close figure
    plt.savefig(save_path)
    plt.close()


def plot_bar_chart(labels, values, title, ylabel, save_path):
    """
    Plot and save a bar chart for model comparison.

    Parameters:
    - labels: List of model names
    - values: Corresponding metric values
    - title: Chart title
    - ylabel: Y-axis label
    - save_path: File path to save the chart
    """
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)

    # Annotate bars with their values
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_curve(history, model_name, save_path):
    """
    Plot training vs validation accuracy over epochs.

    Skips plotting if required keys are missing.
    """
    if "train_accuracy" not in history or "val_accuracy" not in history:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(history, model_name, save_path):
    """
    Plot training vs validation loss over epochs.

    Skips plotting if required keys are missing.
    """
    if "train_loss" not in history or "val_loss" not in history:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_summary_csv(summary_rows, save_path):
    """
    Save model comparison metrics to a CSV file.

    Each row contains:
    Model, Accuracy, Precision, Recall, F1-Score
    """
    with open(save_path, "w") as f:
        f.write("Model,Accuracy,Precision,Recall,F1-Score\n")
        for row in summary_rows:
            f.write(
                f"{row['model']},"
                f"{row['accuracy']:.4f},"
                f"{row['precision']:.4f},"
                f"{row['recall']:.4f},"
                f"{row['f1_score']:.4f}\n"
            )


def main():
    """
    Main execution pipeline.

    Steps:
    1. Load saved model outputs
    2. Compute metrics
    3. Generate plots (confusion matrix, curves, comparisons)
    4. Save results (CSV + JSON)
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_outputs")

    # Create output directory if not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List of models to compare
    models = [
        ("Perceptron", "perceptron"),
        ("MLP", "mlp"),
        ("Improved MLP", "mlp_improved"),
        ("CNN", "cnn"),
    ]

    summary_rows = []

    # Loop through each model
    for model_name, prefix in models:
        y_true_path = os.path.join(RESULTS_DIR, f"{prefix}_y_true.npy")
        y_pred_path = os.path.join(RESULTS_DIR, f"{prefix}_y_pred.npy")
        history_path = os.path.join(RESULTS_DIR, f"{prefix}_history.json")
        metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")

        # Skip model if prediction files are missing
        if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
            print(f"Skipping {model_name}: missing prediction files.")
            continue

        # Load predictions
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)

        # Compute metrics
        results = classification_metrics(y_true, y_pred)
        cm = results["confusion_matrix"]

        # Store summary for comparison charts
        summary_rows.append({
            "model": model_name,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
        })

        # Plot confusion matrix
        plot_confusion_matrix(
            cm,
            f"{model_name} Confusion Matrix",
            os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.png")
        )

        # Plot training curves if history exists
        if os.path.exists(history_path):
            history = load_json(history_path)
            plot_accuracy_curve(
                history,
                model_name,
                os.path.join(OUTPUT_DIR, f"{prefix}_accuracy_curve.png")
            )
            plot_loss_curve(
                history,
                model_name,
                os.path.join(OUTPUT_DIR, f"{prefix}_loss_curve.png")
            )

        # Copy metrics JSON if available
        if os.path.exists(metrics_path):
            metrics_json = load_json(metrics_path)
            with open(os.path.join(OUTPUT_DIR, f"{prefix}_metrics_copy.json"), "w") as f:
                json.dump(metrics_json, f, indent=4)

    # If no valid models found, exit
    if not summary_rows:
        print("No saved model outputs found in the results folder.")
        return

    # Extract values for comparison charts
    model_names = [row["model"] for row in summary_rows]
    accuracies = [row["accuracy"] * 100 for row in summary_rows]
    precisions = [row["precision"] * 100 for row in summary_rows]
    recalls = [row["recall"] * 100 for row in summary_rows]
    f1_scores = [row["f1_score"] * 100 for row in summary_rows]

    # Plot comparison charts
    plot_bar_chart(model_names, accuracies, "Model Test Accuracy Comparison",
                   "Accuracy (%)", os.path.join(OUTPUT_DIR, "model_accuracy_comparison.png"))

    plot_bar_chart(model_names, precisions, "Model Precision Comparison",
                   "Precision (%)", os.path.join(OUTPUT_DIR, "model_precision_comparison.png"))

    plot_bar_chart(model_names, recalls, "Model Recall Comparison",
                   "Recall (%)", os.path.join(OUTPUT_DIR, "model_recall_comparison.png"))

    plot_bar_chart(model_names, f1_scores, "Model F1-Score Comparison",
                   "F1-Score (%)", os.path.join(OUTPUT_DIR, "model_f1_comparison.png"))

    # Save summary metrics
    save_summary_csv(summary_rows, os.path.join(OUTPUT_DIR, "summary_metrics.csv"))

    with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary_rows, f, indent=4)

    print("\nComparison complete.")
    print("Saved outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()