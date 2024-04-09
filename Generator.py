import numpy as np

from ExperimentConfig import ExperimentConfig

class Generator:
    config: ExperimentConfig

    def __init__(self, config):
        self.config = config

    def train(self, x: np.ndarray, y: np.ndarray):
        """Train the generator based on the training data."""
        pass

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data according to config.num_synthetic_shots, config.synth_ratio, and config.num_real_shots."""
        return None, None