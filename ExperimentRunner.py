import json

from ExperimentConfig import ExperimentConfig
from DataLoader import DataLoader
from Generator import Generator
from Classifier import Classifier
from Results import Results
import numpy as np

class ExperimentRunner:
    config: ExperimentConfig
    name: str
    dl: DataLoader

    def __init__(self, config: ExperimentConfig, dl: DataLoader):
        self.config = config
        self.name = config.name
        self.dl = dl

    def run(self) -> Results:
        x_train, y_train, x_test, y_test = self.dl.load_data()

        num_generated = int(self.config.num_real_shots * self.config.sr_ratio)

        num_neg_imgs = int(num_generated / 2)
        num_pos_imgs = num_generated - num_neg_imgs

        generator_neg = Generator(self.config)
        generator_neg.train(x_train[y_train == 0])
        generated_neg = generator_neg.generate(num_neg_imgs)

        generator_pos = Generator(self.config)
        generator_pos.train(x_train[y_train == 1])
        generated_pos = generator_pos.generate(num_pos_imgs)

        generated_x = np.concatenate((generated_neg, generated_pos))
        generated_y = np.concatenate((np.zeros(num_neg_imgs), np.ones(num_pos_imgs)))

        classifier = Classifier(self.config)
        classifier.train(x_train, y_train, generated_x, generated_y)
        return classifier.evaluate(x_test, y_test)
