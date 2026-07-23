import tensorflow as tf
from tensorflow.keras import layers


def build_discriminator(shape=64, channel=3):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    dis = tf.keras.Sequential(name="discriminator")

    dis.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                          input_shape=[shape, shape, channel],
                          kernel_initializer=initializer, use_bias=False))
    dis.add(layers.BatchNormalization())
    dis.add(layers.LeakyReLU())
    dis.add(layers.Dropout(0.2))

    dis.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same",
                          kernel_initializer=initializer, use_bias=False))
    dis.add(layers.BatchNormalization())
    dis.add(layers.LeakyReLU())
    dis.add(layers.Dropout(0.2))

    dis.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                          kernel_initializer=initializer, use_bias=False))
    dis.add(layers.BatchNormalization())
    dis.add(layers.LeakyReLU())
    dis.add(layers.Dropout(0.2))

    dis.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same",
                          kernel_initializer=initializer, use_bias=False))
    dis.add(layers.BatchNormalization())
    dis.add(layers.LeakyReLU())
    dis.add(layers.Dropout(0.2))

    dis.add(layers.Flatten())
    dis.add(layers.Dense(1))

    return dis
