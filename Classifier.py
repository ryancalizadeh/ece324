import numpy as np

from ExperimentConfig import ExperimentConfig
from Results import Results

class Classifier:
    config: ExperimentConfig

    def __init__(self, config):
        self.config = config
    
    def train(self, x: np.ndarray, y: np.ndarray, generated_x: np.ndarray, generated_y: np.ndarray):
        """Train the classifier based on the training data and the generated data."""
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Results:
        """Evaluate the classifier on the test data."""
        return None