"""Define a function to split data into train and test sets."""
from typing import Tuple

import numpy as np


def train_test_split(
    features: np.ndarray, targets: np.ndarray, fraction: float = 0.8
) -> Tuple[np.ndarray, ...]:
    """Split the data into train and test sets.

    Args:
        features: Array of shape (k, n) containing n samples of k features
            each.
        targets: Array of shape (1, n) containing the target values of n
            samples.
        fraction: The fraction of examples to be drawn for training.

    Returns:
        train_features: Subset of features containing n*fraction examples to
            be used for training.
        train_targets: Subset of targets corresponding to train_features
            containing targets.
        test_features: Subset of features containing n - n * fraction examples
            to be used for testing.
        test_targets: Subset of targets corresponding to test_features
            containing targets.
    """
    # Check that the fraction argument is a legal value.
    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("Fraction must be in range [0, 1]")

    train_features: np.ndarray
    test_features: np.ndarray
    train_targets: np.ndarray
    test_targets: np.ndarray
    if fraction == 1.0:
        # If fraction is 1.0, then the test features equals the train features
        train_features = features
        test_features = features
        train_targets = targets
        test_targets = targets
    else:
        # Find indicies to train
        num_train = int(features.shape[1] * fraction)
        examples_to_train = np.sort(
            np.random.choice(a=features.shape[1], size=num_train, replace=False)
        )

        # Break up features and targets
        train_features = features[:, examples_to_train]
        train_targets = targets[:, examples_to_train]
        test_features = np.delete(features, examples_to_train, axis=1)
        test_targets = np.delete(targets, examples_to_train, axis=1)

    return train_features, train_targets, test_features, test_targets
