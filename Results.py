import tensorflow as tf

class Results:
    accuracy: float
    history: tf.keras.callbacks.History

    def __init__(self, accuracy: float, history: tf.keras.callbacks.History):
        self.accuracy = accuracy
        self.history = history