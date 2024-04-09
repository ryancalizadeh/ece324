import ExperimentConfig as ExperimentConfig
import numpy as np

class DataLoader:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data from MedMnist according to config.num_real_shots"""
        return None, None, None, None