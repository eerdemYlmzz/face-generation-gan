import tensorflow as tf
from tensorflow.keras import layers


def build_generator(channel=3, noise_size=100):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    gen = tf.keras.Sequential(name="generator")

    gen.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(noise_size,)))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Reshape((4, 4, 512)))

    gen.add(layers.Conv2DTranspose(512, (3, 3), strides=(1, 1), padding="same",
                                   kernel_initializer=initializer, use_bias=False))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same",
                                   kernel_initializer=initializer, use_bias=False))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same",
                                   kernel_initializer=initializer, use_bias=False))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same",
                                   kernel_initializer=initializer, use_bias=False))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Conv2DTranspose(channel, (5, 5), strides=(2, 2), padding="same",
                                   kernel_initializer=initializer, use_bias=False,
                                   activation="tanh"))

    return gen
