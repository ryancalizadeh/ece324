import json

class ExperimentConfig:
    """
    Config class for running individual experiments.

    config_file: str
        path to the config file

    name: str
        name of the config

    ci_ratio: float
        class imbalance ratio, ratio of the number of positive to negative samples in the real data

    num_real_shots: int
        number of real shots of training data to use

    sr_ratio: float
        ratio of synthetic to real data to use for training the test classifier

    ci_ratio_index: int
        index of the class imbalance ratio in the range of class imbalance ratios

    num_real_shots_index: int
        index of the number of real shots in the range of number of real shots

    sr_ratio_index: int
        index of the synthetic to real ratio in the range of synthetic to real ratios
    """

    config_file: str
    name: str
    ci_ratio: float
    num_real_shots: int
    sr_ratio: float
    ci_ratio_index: int
    num_real_shots_index: int
    sr_ratio_index: int

    def __init__(self, config_file: str):
        self.config_file = config_file
        with open(config_file, "r") as f:
            conf = json.load(f)
            self.name = conf["name"]
            self.ci_ratio = conf["ci_ratio"]
            self.num_real_shots = conf["num_real_shots"]
            self.sr_ratio = conf["sr_ratio"]
            self.ci_ratio_index = conf["ci_ratio_index"]
            self.num_real_shots_index = conf["num_real_shots_index"]
            self.sr_ratio_index = conf["sr_ratio_index"]
