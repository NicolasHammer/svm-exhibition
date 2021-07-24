# Loading packages
import numpy as np


def train_test_split(features: np.ndarray, targets: np.ndarray, fraction: float = 0.8) -> tuple:
    """
    Parameters
    ----------
    features (np.ndarray) - array of shape (k, n) containing n samples of k features each\n
    targets (np.ndarray) - array of shape (1, n) containing the target values of n samples\n
    fraction (float in [0, 1]) - the fraction of examples to be drawn for training

    Output
    ------
    train_features (np.ndarray) - subset of features containing n*fraction examples to be used for training\n
    train_targets (np.ndarray) - subset of targets corresponding to train_features containing targets\n
    test_features (np.ndarray) - subset of features containing n - n*fraction examples to be used for testing\n
    test_targets (np.ndarray) - subset of targets corresponding to test_features containing targets
    """
    # Edge cases
    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("Fraction must be in range [0, 1]")
    elif fraction == 1.0:  # edge case where test_features = train_features
        train_features = features
        test_features = features
        train_targets = targets
        test_targets = targets
        return train_features, train_targets, test_features, test_targets

    # Main case
    # Find indicies to train
    num_train = int(features.shape[1]*fraction)
    examples_to_train = np.sort(np.random.choice(
        a=features.shape[1], size=num_train, replace=False))

    # Break up features and targets
    train_features = features[:, examples_to_train]
    train_targets = targets[:, examples_to_train]
    test_features = np.delete(features, examples_to_train, axis=1)
    test_targets = np.delete(targets, examples_to_train, axis=1)

    return train_features, train_targets, test_features, test_targets