import json

class ExperimentConfig:
    """
    Config class for running individual experiments.

    config_file: str
        path to the config file

    name: str
        name of the config

    sr_ratio: float
        ratio of synthetic to real data to use for training the test classifier
    
    num_real_shots: int
        number of real shots of training data to use

    ci_ratio: float
        class imbalance ratio, ratio of the number of positive to negative samples in the real data
    """

    config_file: str
    name: str
    sr_ratio: float
    num_real_shots: int
    ci_ratio: float

    def __init__(self, config_file: str):
        self.config_file = config_file
        with open(config_file, "r") as f:
            conf = json.load(f)
            self.name = conf["name"]
            self.sr_ratio = conf["sr_ratio"]
            self.num_real_shots = conf["num_real_shots"]
            self.ci_ratio = conf["ci_ratio"]
