import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ExperimentConfig import ExperimentConfig

class Generator:
    config: ExperimentConfig

    def __init__(self, config):
        self.config = config
        self.generator = None
        self.discriminator = None
        
    def dcgan_generator(self):
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
        print(model.output_shape)
        assert model.output_shape == (None, 28, 28, 1)
    
        return model

    def dcgan_discriminator(self):
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

    def generator_loss(fake_out, loss):
        return loss(tf.ones_like(fake_output), fake_output)
        
    def discriminator_loss(real_out, fake_out, loss):
        real_loss = loss(tf.ones_like(real_output), real_output)
        fake_loss = loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
        
    def train(self, x: np.ndarray, y: np.ndarray):
        self.generator = dcgan_generator(config)
        self.discriminator = dcgan_discriminator(config)
        """Train the generator based on the training data."""
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) # maybe specify in config
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        checkpoint_dir = './gan_training_checkpoints' # maybe move to config
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
        
        EPOCHS = 50
        noise_dim = 100

        for epoch in range(epochs):
            for image_batch in dataset:
                noise = tf.random.normal([BATCH_SIZE, noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(noise, training=True)
                    
                    real_output = discriminator(images, training=True)
                    fake_output = discriminator(generated_images, training=True)
                    
                    gen_loss = generator_loss(fake_output, loss)
                    disc_loss = discriminator_loss(real_output, fake_output, loss)
            
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
                    
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
              checkpoint.save(file_prefix = checkpoint_prefix)

        return model
                
    def generate(self, num_images) -> np.ndarray:
        """Generate synthetic data according to config.num_synthetic_shots, config.synth_ratio, and config.num_real_shots."""
        generator = self.generator
        num_gen_imgs = self.config["num_synthetic_shots"]
        s_ratio = self.config["synth_ratio"]
        class_imbalance = self.config["ci_ratio"]
        seed = tf.random.normal([num_images, 100])
        generated_images = generator(seed, training=False)
        return generated_images
