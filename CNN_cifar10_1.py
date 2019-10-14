from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt


class SimpleNetwork:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # 全连接部分
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # 输出层
        model.add(Dense(NB_CLASSES))
        model.add(Activation('softmax'))
        return model


class ComplexNetWork:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, kernel_size=3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(513))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model


# 图片基本信息，CIFAR-10是一个包含了60000张32*32大小的三通道图像数据集
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)

# 基本参数
NB_EPOCH = 40
BATCH_SIZE = 128
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = RMSprop()

# 导入数据，加载数据集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(y_train.shape)
print('X_train shape', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# 对输出结果y进行分类转换
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 归一化处理
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 设计网络

# model = SimpleNetwork.build(INPUT_SHAPE, NB_CLASSES)
model = ComplexNetWork.build(INPUT_SHAPE, NB_CLASSES)
model.summary()     # 输出模型情况
# 编译网络并训练
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.fit(X_train, y_train, verbose=VERBOSE, epochs=NB_EPOCH, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])

# 保存模型
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
# 保存权重
model.save_weights('cifar10_weights.h5', overwrite=True)
