import json

from ExperimentConfig import ExperimentConfig
from DataLoader import DataLoader
from Generator import Generator
from Classifier import Classifier
from Results import Results

class ExperimentRunner:
    config: ExperimentConfig

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.name = config.name

    def run(self) -> Results:
        dl = DataLoader(self.config)
        train_x, train_y, test_x, test_y = dl.load_data()

        generator = Generator(self.config)
        generator.train(train_x, train_y)
        generated_x, generated_y = generator.generate()

        classifier = Classifier(self.config)
        classifier.train(train_x, train_y, generated_x, generated_y)
        return classifier.evaluate(test_x, test_y)
