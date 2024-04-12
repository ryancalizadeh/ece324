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
        train_gen, train_x, train_y, test_x, test_y = self.dl.load_data()

        num_imgs = self.config.num_real_shots
        s_ratio = self.config.sr_ratio
        class_imbalance = self.config.ci_ratio
        num_synth_imgs = int(num_imgs * s_ratio)
        
        generator_0 = Generator(self.config)
        generator_0.train(train_gen[1], train_y)
        num_neg_imgs = int(num_synth_imgs * (1-class_imbalance))
        generated_neg = generator_0.generate(num_neg_imgs)

        generator_1 = Generator(self.config)
        generator_1.train(train_gen[0], train_y)
        num_pos_imgs = num_synth_imgs - num_neg_imgs
        generated_pos = generator_1.generate()

        generated_x = np.concatenate(generated_neg, generated_pos)
        generated_y = np.concatenate(np.zeros(num_neg_imgs), np.ones(num_pos_imgs))
        
        classifier = Classifier(self.config)
        classifier.train(train_x, train_y, generated_x, generated_y)
        return classifier.evaluate(test_x, test_y)
