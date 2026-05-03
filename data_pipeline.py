"""
Author: Manu Saini (7092158) and Kev Akpinar (7159320)
Course: Machine Learning Project 4
Task: CelebA facial attribute classification (Smiling)

This file implements the dataset loading and preprocessing pipeline for Project 4.
It supports:
1. Building the dataset from raw CelebA images
2. Shuffling and splitting the dataset
3. Saving the processed arrays as .npy files
4. Loading cached .npy files to avoid rebuilding every run
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class CelebADataLoader:
    """
    Loads CelebA images and the selected binary attribute.

    This loader prepares two forms of input:
    1. Image tensors for the CNN: shape (N, H, W, C)
    2. Flattened feature vectors for perceptron/MLP: shape (N, H*W*C)
    """

    def __init__(
        self,
        image_dir: str,
        attr_file: str,
        target_attribute: str = "Smiling",
        image_size: Tuple[int, int] = (32, 32),
        seed: int = 42,
    ) -> None:
        self.image_dir = image_dir
        self.attr_file = attr_file
        self.target_attribute = target_attribute
        self.image_size = image_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def load_attribute_labels(self) -> Tuple[List[str], List[int]]:
        """
        Read the CelebA attribute CSV file and extract labels for the target attribute.

        Returns:
            image_names: list of image filenames
            labels: list of integer labels in {0, 1}
        """
        if not os.path.exists(self.attr_file):
            raise FileNotFoundError(f"Attribute file not found: {self.attr_file}")

        image_names: List[str] = []
        labels: List[int] = []

        with open(self.attr_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError("CSV attribute file is empty or invalid.")

        header = lines[0].strip().split(",")
        if self.target_attribute not in header:
            raise ValueError(
                f"Target attribute '{self.target_attribute}' not found in CSV file."
            )

        image_col = header.index("image_id")
        target_col = header.index(self.target_attribute)

        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue

            filename = parts[image_col]
            raw_label = int(parts[target_col])
            binary_label = 1 if raw_label == 1 else 0

            image_names.append(filename)
            labels.append(binary_label)

        return image_names, labels

    def load_single_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess one RGB image.

        Steps:
            1. Open image
            2. Convert to RGB
            3. Resize to image_size
            4. Normalize pixels to [0, 1]

        Returns:
            NumPy array of shape (H, W, 3), dtype float32
        """
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return image_array

    def build_dataset(
        self,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load images and labels into NumPy arrays.

        Args:
            max_samples: optional limit for faster debugging

        Returns:
            X_images: shape (N, H, W, C)
            X_flat:   shape (N, H*W*C)
            y:        shape (N, 1)
        """
        image_names, labels = self.load_attribute_labels()

        X_images: List[np.ndarray] = []
        y: List[int] = []

        total_available = len(image_names)
        limit = total_available if max_samples is None else min(max_samples, total_available)

        for i in range(limit):
            filename = image_names[i]
            label = labels[i]
            image_path = os.path.join(self.image_dir, filename)

            if not os.path.exists(image_path):
                continue

            try:
                image_array = self.load_single_image(image_path)
                X_images.append(image_array)
                y.append(label)
            except Exception as exc:
                print(f"Warning: failed to load {filename}: {exc}")

        if not X_images:
            raise ValueError("No images were loaded. Check the image directory and file paths.")

        X_images_np = np.array(X_images, dtype=np.float32)
        X_flat_np = X_images_np.reshape(X_images_np.shape[0], -1)
        y_np = np.array(y, dtype=np.float32).reshape(-1, 1)

        return X_images_np, X_flat_np, y_np

    def shuffle_dataset(
        self,
        X_images: np.ndarray,
        X_flat: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shuffle all arrays using the same permutation.
        """
        n_samples = X_images.shape[0]
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)

        return X_images[indices], X_flat[indices], y[indices]

    def split_dataset(
        self,
        X_images: np.ndarray,
        X_flat: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Dict[str, np.ndarray]:
        """
        Split the dataset into train, validation, and test sets.

        Returns a dictionary with both image and flattened inputs.
        """
        n_samples = X_images.shape[0]
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)

        splits = {
            "X_train_img": X_images[:train_end],
            "X_val_img": X_images[train_end:val_end],
            "X_test_img": X_images[val_end:],
            "X_train_flat": X_flat[:train_end],
            "X_val_flat": X_flat[train_end:val_end],
            "X_test_flat": X_flat[val_end:],
            "y_train": y[:train_end],
            "y_val": y[train_end:val_end],
            "y_test": y[val_end:],
        }
        return splits

    @staticmethod
    def summarize_dataset(X_images: np.ndarray, X_flat: np.ndarray, y: np.ndarray) -> None:
        """
        Print a simple dataset summary.
        """
        total_samples = X_images.shape[0]
        positive_count = int(np.sum(y))
        negative_count = total_samples - positive_count

        print("=" * 60)
        print("CelebA Dataset Summary")
        print("=" * 60)
        print(f"Total samples loaded     : {total_samples}")
        print(f"Image tensor shape       : {X_images.shape}")
        print(f"Flattened tensor shape   : {X_flat.shape}")
        print(f"Positive labels (1)      : {positive_count}")
        print(f"Negative labels (0)      : {negative_count}")
        print("=" * 60)

    @staticmethod
    def save_splits(splits: Dict[str, np.ndarray], save_dir: str) -> None:
        """
        Save all dataset splits as .npy files.
        """
        os.makedirs(save_dir, exist_ok=True)

        for name, array in splits.items():
            path = os.path.join(save_dir, f"{name}.npy")
            np.save(path, array)

        print(f"Saved processed dataset to: {save_dir}")

    @staticmethod
    def load_splits(save_dir: str) -> Dict[str, np.ndarray]:
        """
        Load all dataset splits from .npy files.
        """
        required_files = [
            "X_train_img.npy",
            "X_val_img.npy",
            "X_test_img.npy",
            "X_train_flat.npy",
            "X_val_flat.npy",
            "X_test_flat.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test.npy",
        ]

        missing_files = [
            filename for filename in required_files
            if not os.path.exists(os.path.join(save_dir, filename))
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Missing cached files in {save_dir}: {missing_files}"
            )

        splits = {
            "X_train_img": np.load(os.path.join(save_dir, "X_train_img.npy")),
            "X_val_img": np.load(os.path.join(save_dir, "X_val_img.npy")),
            "X_test_img": np.load(os.path.join(save_dir, "X_test_img.npy")),
            "X_train_flat": np.load(os.path.join(save_dir, "X_train_flat.npy")),
            "X_val_flat": np.load(os.path.join(save_dir, "X_val_flat.npy")),
            "X_test_flat": np.load(os.path.join(save_dir, "X_test_flat.npy")),
            "y_train": np.load(os.path.join(save_dir, "y_train.npy")),
            "y_val": np.load(os.path.join(save_dir, "y_val.npy")),
            "y_test": np.load(os.path.join(save_dir, "y_test.npy")),
        }

        print(f"Loaded processed dataset from: {save_dir}")
        return splits

    @staticmethod
    def cache_exists(save_dir: str) -> bool:
        """
        Check whether all cached .npy split files exist.
        """
        required_files = [
            "X_train_img.npy",
            "X_val_img.npy",
            "X_test_img.npy",
            "X_train_flat.npy",
            "X_val_flat.npy",
            "X_test_flat.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test.npy",
        ]

        return all(
            os.path.exists(os.path.join(save_dir, filename))
            for filename in required_files
        )


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    IMAGE_DIR = os.path.join(BASE_DIR, "Data", "img_align_celeba", "img_align_celeba")
    ATTR_FILE = os.path.join(BASE_DIR, "data", "list_attr_celeba.csv")
    CACHE_DIR = os.path.join(BASE_DIR, "processed_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    MAX_SAMPLES = 6000
    FORCE_REBUILD = False

    print("Looking for images in:", IMAGE_DIR)
    print("Looking for attribute file in:", ATTR_FILE)
    print("Cache directory:", CACHE_DIR)

    loader = CelebADataLoader(
        image_dir=IMAGE_DIR,
        attr_file=ATTR_FILE,
        target_attribute="Smiling",
        image_size=(32, 32),
        seed=42,
    )

    if not FORCE_REBUILD and loader.cache_exists(CACHE_DIR):
        print("\nCached dataset found. Loading .npy files instead of rebuilding from images...\n")
        splits = loader.load_splits(CACHE_DIR)

    else:
        print("\nNo cache found, or FORCE_REBUILD=True. Building dataset from images...\n")
        X_images, X_flat, y = loader.build_dataset(max_samples=MAX_SAMPLES)
        loader.summarize_dataset(X_images, X_flat, y)

        X_images, X_flat, y = loader.shuffle_dataset(X_images, X_flat, y)
        splits = loader.split_dataset(X_images, X_flat, y)

        loader.save_splits(splits, CACHE_DIR)

    print("Train flat split:", splits["X_train_flat"].shape, splits["y_train"].shape)
    print("Validation flat split:", splits["X_val_flat"].shape, splits["y_val"].shape)
    print("Test flat split:", splits["X_test_flat"].shape, splits["y_test"].shape)

    print("Train image split:", splits["X_train_img"].shape)
    print("Validation image split:", splits["X_val_img"].shape)
    print("Test image split:", splits["X_test_img"].shape)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_summary = {
        "target_attribute": "Smiling",
        "max_samples": MAX_SAMPLES,
        "train_flat_shape": list(splits["X_train_flat"].shape),
        "val_flat_shape": list(splits["X_val_flat"].shape),
        "test_flat_shape": list(splits["X_test_flat"].shape),
        "train_img_shape": list(splits["X_train_img"].shape),
        "val_img_shape": list(splits["X_val_img"].shape),
        "test_img_shape": list(splits["X_test_img"].shape),
        "y_train_shape": list(splits["y_train"].shape),
        "y_val_shape": list(splits["y_val"].shape),
        "y_test_shape": list(splits["y_test"].shape),
        "train_positive": int(np.sum(splits["y_train"])),
        "train_negative": int(splits["y_train"].shape[0] - np.sum(splits["y_train"])),
        "val_positive": int(np.sum(splits["y_val"])),
        "val_negative": int(splits["y_val"].shape[0] - np.sum(splits["y_val"])),
        "test_positive": int(np.sum(splits["y_test"])),
        "test_negative": int(splits["y_test"].shape[0] - np.sum(splits["y_test"])),
    }

    with open(os.path.join(RESULTS_DIR, "data_summary.json"), "w") as f:
        json.dump(data_summary, f, indent=4)

    print("Data summary saved to:", os.path.join(RESULTS_DIR, "data_summary.json"))