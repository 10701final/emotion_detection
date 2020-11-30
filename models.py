from keras.layers import Convolution2D, experimental, BatchNormalization, Dropout, Dense, Activation, MaxPooling2D, \
    Flatten
from keras import Sequential


def base_model():
    base_model = Sequential([
        experimental.preprocessing.Rescaling(1. / 255),

        Convolution2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
        Convolution2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'),
        Dropout(0.4),

        Convolution2D(128, (3, 3), padding='same'),
        Convolution2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'),
        Dropout(0.4),

        Convolution2D(256, (3, 3), padding='same'),
        Convolution2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'),
        Dropout(0.4),

        Convolution2D(512, (3, 3), padding='same'),
        Convolution2D(512, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'),
        Dropout(0.4),

        Flatten(),

        Dense(2048),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        Dense(7),
        Activation('softmax'),
    ])
    return base_model
