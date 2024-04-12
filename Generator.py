import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from ExperimentConfig import ExperimentConfig
import os 

class Generator:
    config: ExperimentConfig
    generator: keras.Sequential
    discriminator: keras.Sequential

    def __init__(self, config):
        self.config = config
        self.generator = Generator.dcgan_generator()
        self.discriminator = Generator.dcgan_discriminator()
    
    @staticmethod
    def dcgan_generator():
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Reshape((7, 7, 256)))
    
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)
    
        return model

    @staticmethod
    def dcgan_discriminator():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def generator_loss(self, fake_out, loss):
        return loss(tf.ones_like(fake_out), fake_out)
        
    def discriminator_loss(self, real_out, fake_out, loss):
        real_loss = loss(tf.ones_like(real_out), real_out)
        fake_loss = loss(tf.zeros_like(fake_out), fake_out)
        total_loss = real_loss + fake_loss
        return total_loss
        
    def train(self, x: np.ndarray):
        """Train the generator based on the training data."""
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        EPOCHS = 50
        noise_dim = 100
        dataset = tf.data.Dataset.from_tensor_slices(x)
        N = len(x)

        for epoch in range(EPOCHS):
            for image_batch in dataset:
                noise = tf.random.normal([N, noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.generator(noise, training=True)
                    
                    real_output = self.discriminator(image_batch, training=True)
                    fake_output = self.discriminator(generated_images, training=True)
                    
                    gen_loss = self.generator_loss(fake_output, loss)
                    disc_loss = self.discriminator_loss(real_output, fake_output, loss)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            
                generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def generate(self, num_images) -> np.ndarray:
        """Generate synthetic data according to config.num_synthetic_shots, config.synth_ratio, and config.num_real_shots."""
        seed = tf.random.normal([num_images, 100])
        generated_images = self.generator(seed, training=False)
        return generated_images
