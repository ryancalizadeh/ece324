from ExperimentConfig import ExperimentConfig
import numpy as np

from medmnist import PneumoniaMNIST
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DataLoader:
    x: np.ndarray
    y: np.ndarray
    x_pos: np.ndarray
    x_neg: np.ndarray

    config: ExperimentConfig | None

    def __init__(self, config: ExperimentConfig | None=None):
        self.config = config

        train_dataset = PneumoniaMNIST(split='train', download=True, transform=lambda im : np.array(im.getdata()).reshape(im.size[0], im.size[1]))
        test_dataset = PneumoniaMNIST(split='test', download=True, transform=lambda im : np.array(im.getdata()).reshape(im.size[0], im.size[1]))
        val_dataset = PneumoniaMNIST(split='val', download=True, transform=lambda im : np.array(im.getdata()).reshape(im.size[0], im.size[1]))

        x_train, y_train = self.dataset_to_numpy(train_dataset)
        x_test, y_test = self.dataset_to_numpy(test_dataset)
        x_val, y_val = self.dataset_to_numpy(val_dataset)

        x_train = DataLoader.normalize(x_train)
        x_test = DataLoader.normalize(x_test)
        x_val = DataLoader.normalize(x_val)

        y_train = y_train[:, 0]
        y_test = y_test[:, 0]
        y_val = y_val[:, 0]

        # Combine datasets
        self.x = np.concatenate((x_train, x_test, x_val))
        self.y = np.concatenate((y_train, y_test, y_val))

        self.x_pos = self.x[self.y == 1]
        self.x_neg = self.x[self.y == 0]

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from MedMnist according to config.num_real_shots and config.ci_ratio.
        returns: x_train, y_train, x_test, y_test
        """

        assert self.config is not None

        num_pos = int(self.config.num_real_shots * self.config.ci_ratio)
        num_neg = self.config.num_real_shots - num_pos

        rng = np.random.default_rng()
        x_pos = rng.choice(self.x_pos, num_pos, replace=False)
        x_neg = rng.choice(self.x_neg, num_neg, replace=False)

        x_train = np.concatenate((x_pos, x_neg))
        y_train = np.concatenate((np.ones(num_pos), np.zeros(num_neg)))

        # Shuffle the data
        shuffle(x_train, y_train)

        # Optimization so we aren't testing on a massive dataset
        _, x_test, _, y_test = train_test_split(self.x, self.y, test_size=1000)

        return x_train, y_train, x_test, y_test

    def set_config(self, config):
        self.config = config

    def dataset_to_numpy(self, dataset):
        x = []
        y = []
        for i in range(len(dataset)):
            x.append(dataset[i][0])
            y.append(dataset[i][1])
        return np.array(x), np.array(y)

    @staticmethod
    def normalize(x):
        x = 2 * x / np.max(x) - 1
        return x
