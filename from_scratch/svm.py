import numpy as np


class SVM():
    def __init__(self, k: int, max_epochs: int = 5000, reg_strength: float = 10000,
                 learning_rate: float = 0.000001, cost_threshold: float = 0.01):
        """
        Member variables
        ----------------
        k (int) - the number of weights in the weight array
        """
        self.weights = np.zeros((1, k))
        self.reg_strength = reg_strength
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.cost_threshold = cost_threshold

    def fit(self, data):
        prev_cost = float("inf")
        for epoch in range(self.max_epochs):
            X, Y = 


    def predict(self, data):
        pass

    def compute_cost(self, features: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the loss of the SVM.

        Parameters
        ----------
        features (np.ndarray) - array of shape (k, n) containing n samples of k features each\n
        targets (np.ndarray) - array of shape (1, n) containing the target values of n samples\n

        Output
        ------
        cost (float) - hinge loss + L2 regularization 
        """
        distances = 1 - targets * \
            np.matmul(self.weights, features)  # (1, n) + (1, k) x (k, n)
        distances[distances < 0] = 0

        hinge_loss = self.reg_strength * np.sum(distances) / features.shape[1]

        cost = 0.5 * np.matmul(self.weights, self.weights.T) + hinge_loss
        return cost

    def compute_gradient(self, features: np.ndarray, targets: np.ndarray) -> float:
        """
        Evaluate the gradient of the loss function.

        Parameters
        ----------
        features (np.ndarray) - array of shape (k, n) containing n samples of k features each\n
        targets (np.ndarray) - array of shape (1, n) containing the target values of n samples\n

        Output
        ------
        delta (float) - gradient of the loss function
        """
        distances = 1 - targets * \
            np.matmul(self.weights, features)  # (1, n) + (1, k) x (k, n)
        zeroed_out_samples = features[:, distances < 0] = 0

        delta = self.weights - self.reg_strength * \
            np.matmul(targets, zeroed_out_samples.T)  # (1, k) - (1, n) x (n, k)
        return delta
