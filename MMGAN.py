from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers.instancenormalization import InstanceNormalization
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt


class Generative:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        model.add(Conv2DTranspose(4, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(64, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(128, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(256, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(512, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(512, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(512, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2DTranspose(512, kernel_size=4, padding='same', input_shape=input_shape))
        model.add(InstanceNormalization(axis=None))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2D(512, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(1024, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(1024, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(1024, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(1024, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(512, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        model.add(Conv2D(4, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.2))
        return model


class Discriminator:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        model.add(Conv2D(8,kernel_size=4, input_shape=input_shape, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(128, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(256, kernel_size=4, padding='same'))
        model.add(InstanceNormalization(axis=None))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(4, kernel_size=16, padding='same'))
        return model


input_shape = (256, 256, 4)
model = Discriminator.build(input_shape)
model.summary()



