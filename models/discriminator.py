
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, LeakyReLU, \
         BatchNormalization, Embedding, Concatenate, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_math_ops import Mod


def create_discriminator(config):

    img_input = Input(shape=config.image_shape, name='image input')

    img_layer = Conv2D(8, (16,16), strides=(1,1), padding='same', input_shape=config.image_shape)(img_input)
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    # img_layer = (MaxPool2D(pool_size=(2,2))) #shape -> 200,200,8 -> 100,100,8
    img_layer = Conv2D(32, (8,8), strides=(2,2), padding='same')(img_layer) #shape -> 100,100,8 -> 100,100,32
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    # img_layer = (MaxPool2D(pool_size=(2,2))) #shape -> 100,100,32 -> 50,50,32
    img_layer = Conv2D(32, (4,4), strides=(2,2), padding='same')(img_layer)
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    # img_layer = (MaxPool2D(pool_size=(2,2))) #shape -> 50,50,32 -> 25,25,32
    img_layer = Conv2D(32, (4,4), strides=(2,2), padding='same')(img_layer)
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Flatten()(img_layer)
    img_layer = Dense(64)(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)
    img_layer = Dense(64)(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)
    img_layer = Dense(64)(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    out_layer = Dense(1, activation='sigmoid')(img_layer)

    model = Model(img_input, out_layer, name='discriminator')

    return model


def discriminator_loss(real_output, fake_output):
    cross_entropy = BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss