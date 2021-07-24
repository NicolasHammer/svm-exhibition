import numpy as np


class SVM():

    def __init__(self, k: int, max_epochs: int = 5000, reg_strength: float = 10000,
                 learning_rate: float = 0.000001, cost_threshold: float = 0.01):
        """
        A support vector machine that finds the best linear classifier between two classes.

        Member variables
        ----------------
        k (int) - the number of weights in the weight array
        """
        self.weights = np.zeros((1, k))
        self.reg_strength = reg_strength
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.cost_threshold = cost_threshold

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Learn the weights of the model based on features and targets"""
        X = np.copy(features)
        Y = np.copy(targets)
        prev_cost = float("inf")
        nth = 0
        shuffle_array = np.arange(X.shape[1])

        for epoch in range(self.max_epochs):
            np.random.shuffle(shuffle_array)
            X = X[:, shuffle_array]
            Y = Y[:, shuffle_array]

            # Stochastic gradient descent
            for i, sample in enumerate(X.T):
                grad = self.compute_gradient(sample.reshape(
                    sample.shape[0], 1), np.atleast_1d(Y[0, i]).reshape(1, 1))
                self.weights = self.weights - self.learning_rate * grad

            # convergence check on 2^nth epoch
            if epoch == 2**nth or epoch == self.max_epochs - 1:
                cost = self.compute_cost(X, Y)
                print(f"| epoch {epoch} | loss {cost}")
                if abs(prev_cost - cost) < self.cost_threshold * prev_cost:
                    break
                else:
                    prev_cost = cost
                    nth += 1

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the labels of samples using the trained model"""
        return np.sign(np.matmul(self.weights, data))

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
        distances[:, distances.flatten() < 0] = 0

        hinge_loss = self.reg_strength * np.sum(distances) / features.shape[1]

        cost = 0.5 * np.matmul(self.weights,
                               self.weights.T).item() + hinge_loss
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
        distances = (1 - targets * \
                     np.matmul(self.weights, features)).T  # (1, n) + (1, k) x (k, n)

        delta = self.weights - self.reg_strength * \
            np.matmul(targets, features.T*(distances >= 0))  # (1, k) - (1, n) x (n, k)
        return delta
