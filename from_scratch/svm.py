import numpy as np

class SVM():
    def __init__(self):
        self.weights = np.ndarray()
        self.reg_strength: float

    def fit(self, data):
        pass

    def predict(self, data):
        pass

    def compute_cost(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss of the SVM"""
        N = features.shape[1]
        distances = 1 - targets * np.dot(self.weights, features)
        distances[distances < 0] = 0
