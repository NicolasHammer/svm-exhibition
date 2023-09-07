"""Define the support vector machine class."""
import logging

import numpy as np


class SupportVectorMachine():
    """A support vector machine used for binary classification."""

    weights: np.ndarray
    reg_strength: float
    max_epochs: int
    learning_rate: float
    cost_threshold: float

    def __init__(
        self, 
        k: int, 
        max_epochs: int = 5000, 
        reg_strength: float = 10000,
        learning_rate: float = 0.000001, 
        cost_threshold: float = 0.01
    ) -> None:
        """
        A support vector machine that finds the best linear classifier between
        two classes.

        Args:
            k: The number of weights in the weight array.
            max_epochs: The maximum number of epochs to train the SVM.
            reg_strength: The strength of the Hinge loss's reguralization.
            learning_rate: The rate at which the SVM learns.
            cost_threshold: The proporition of the most recent cost which the 
                difference in costs must be under to satisfy convergence.
        """
        self.weights = np.zeros((1, k))
        self.reg_strength = reg_strength
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.cost_threshold = cost_threshold

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Learn the weights of the model based on features and targets.
        
        Args:
            features: The features of the model of shape (k, n) containing n
                samples of k features each.
            targets: The targets of the model with n samples of shape (1, n) 
                containing the target values of n samples.
        """
        prev_cost = float("inf")
        nth: int = 0
        shuffle_array: np.ndarray = np.arange(features.shape[1])
        for epoch in range(self.max_epochs):
            # Shuffle the feature and target arrays
            np.random.shuffle(shuffle_array)
            shuffled_features: np.ndarray = features[:, shuffle_array]
            shuffled_targets: np.ndarray = targets[:, shuffle_array]

            # Stochastic gradient descent
            i: int
            sample: np.ndarray
            for i, sample in enumerate(shuffled_features.T):
                grad: float = self.compute_gradient(
                    sample.reshape(sample.shape[0], 1), 
                    np.atleast_1d(shuffled_targets[0, i]).reshape(1, 1)
                )
                self.weights = self.weights - self.learning_rate * grad

            # Convergence check on 2^nth or final epoch
            if (epoch == 2**nth) or (epoch == self.max_epochs - 1):
                cost: float = self.compute_cost(shuffled_features, shuffled_targets)
                logging.info(f"| epoch {epoch} | loss {cost} |")
                if abs(prev_cost - cost) < self.cost_threshold * prev_cost:
                    break
                else:
                    prev_cost = cost
                    nth += 1

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the labels of samples using the trained model.
        
        Args:
            data: An array of shape (k, n) containing n samples of k features 
                each to predict labels from.

        Returns:
            An array of shape (1, n) of the predicted labels.
        """
        return np.sign(np.matmul(self.weights, data))

    def compute_cost(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss of the SVM.

        Args:
            features: An array of shape (k, n) containing n samples of k 
                features each.
            targets: An array of shape (1, n) containing the target values of 
                n samples.

        Returns:
            Hinge loss + L2 regularization.
        """
        distances: np.ndarray = 1 - targets * \
            np.matmul(self.weights, features)  # (1, n) + (1, k) x (k, n)
        distances[:, distances.flatten() < 0] = 0

        hinge_loss: float = (
            self.reg_strength * np.sum(distances) / features.shape[1]
        )

        cost: float = (
            0.5 * np.matmul(self.weights, self.weights.T).item() + hinge_loss
        )

        return cost

    def compute_gradient(
        self, features: np.ndarray, targets: np.ndarray
    ) -> float:
        """Evaluate the gradient of the loss function.

        Args:
            features: An array of shape (k, n) containing n samples of k
                features each
            targets: An array of shape (1, n) containing the target values of 
                n samples

        Returns:
            Gradient of the loss function.
        """
        distances = (
            (1 - targets * np.matmul(self.weights, features)).T
        )  # (1, n) - (1, k) x (k, n)

        delta = (
            self.weights - self.reg_strength * np.matmul(
                targets, features.T*(distances >= 0)
            )
        )  # (1, k) - (1, n) x (n, k)
        return delta
