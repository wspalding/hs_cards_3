
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU, \
         BatchNormalization, Embedding, Concatenate, Input, Reshape
from tensorflow.keras import initializers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import optimizers


def create_generator(config):
    random_dim = config.generator_seed_dim

    noise_input = Input(shape=(random_dim,), name='noise input')

    seed_layer = Dense(256, input_dim=random_dim, kernel_initializer=tf.random_normal_initializer(stddev=0.02))(noise_input)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Dense(512)(seed_layer)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Dense(1024)(seed_layer)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Dense(1875)(seed_layer)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Reshape((25,25, 3))(seed_layer)

    img_layer = Conv2DTranspose(16, (10,10), strides=(2,2), padding='same')(seed_layer) # shape -> 50,50,3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Conv2DTranspose(32, (7,7), strides=(2,2), padding='same')(img_layer) # shape -> 100, 100, 3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Conv2DTranspose(32, (5,5), strides=(2,2), padding='same')(img_layer) # shape -> 200, 200, 3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Conv2DTranspose(32, (5,5), strides=(1,1), padding='same')(img_layer) # shape -> 200, 200, 3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    out_layer = Conv2DTranspose(3, (3,3), strides=(1,1), padding='same', activation='tanh')(img_layer)

    model = Model(noise_input, out_layer, name='generator')

    return model


def generator_loss(fake_output):
    cross_entropy = BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)