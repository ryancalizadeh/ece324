import json

from ExperimentConfig import ExperimentConfig
from DataLoader import DataLoader
from Generator import Generator
from Classifier import Classifier
from Results import Results

class ExperimentRunner:
    config: ExperimentConfig
    name: str
    dl: DataLoader

    def __init__(self, config: ExperimentConfig, dl: DataLoader):
        self.config = config
        self.name = config.name
        self.dl = dl

    def run(self) -> Results:
        train_x, train_y, test_x, test_y = self.dl.load_data()

        generator = Generator(self.config)
        generator.train(train_x, train_y)
        generated_x, generated_y = generator.generate()

        classifier = Classifier(self.config)
        classifier.train(train_x, train_y, generated_x, generated_y)
        return classifier.evaluate(test_x, test_y)
