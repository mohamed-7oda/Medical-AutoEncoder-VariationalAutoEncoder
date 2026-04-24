import tensorflow as tf
from tensorflow.keras import layers, Model

def build_ae(latent_dim=32):
    encoder_inputs = layers.Input(shape=(64, 64, 1))
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim)(x)

    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, outputs)

    outputs = decoder(latent)
    autoencoder = Model(encoder_inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder
