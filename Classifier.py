import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from ExperimentConfig import ExperimentConfig
from Results import Results

class Classifier:
    config: ExperimentConfig
    epochs: int
    history: tf.keras.callbacks.History

    def __init__(self, config, epochs=10):
        self.config = config
        self.epochs = epochs

    def define_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train(self, x: np.ndarray, y: np.ndarray, generated_x: np.ndarray, generated_y: np.ndarray):
        """Train the classifier based on the training data and the generated data."""

        # combine real and generated

        generated_x = np.squeeze(generated_x, axis=-1)

        if len(generated_y.shape) > 1:
            generated_y = np.squeeze(generated_y, axis=-1)
        
        combined_x = np.concatenate((x, generated_x), axis=0)
        combined_y = np.concatenate((y, generated_y), axis=0)

        # shuffle the combined data
        combined_x, combined_y = shuffle(combined_x, combined_y)
        combined_x = tf.convert_to_tensor(np.array(combined_x).reshape(-1, 28, 28, 1).astype(np.float32))
        combined_y = tf.convert_to_tensor(np.array(combined_y).astype(np.float32))
 
        model = self.define_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.history = model.fit(combined_x, combined_y, epochs=self.epochs)

        self.model = model

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Results:
        """Evaluate the classifier on the test data."""

        # evaluate
        loss, accuracy = self.model.evaluate(x, y)

        results = Results(accuracy, self.history)

        return results
