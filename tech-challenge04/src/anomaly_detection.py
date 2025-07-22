import numpy as np

class AnomalyDetector:
    def __init__(self, threshold=15.0, window_size=5):
        self.threshold = threshold
        self.window_size = window_size
        self.history = []

    def detect(self, keypoints):
        if keypoints is None or not isinstance(keypoints, (list, np.ndarray)):
            return False

        keypoints = np.array(keypoints)
        if np.isnan(keypoints).any():
            return False

        self.history.append(keypoints)

        if len(self.history) < self.window_size + 1:
            return False

        # Cálculo da média do histórico anterior
        prev_mean = np.mean(self.history[-self.window_size - 1:-1], axis=0)
        distance = np.linalg.norm(keypoints - prev_mean)

        return distance > self.threshold
