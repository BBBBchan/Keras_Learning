from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras import backend as k
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt


class LeNet:    # 建立LeNet类，封装网络结构
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # 构建卷积层
        model.add(Conv2D(20, kernel_size=5,
                         padding="same", input_shape=input_shape))   # 输入层接入，20个5*5滤波器，输出维度和输入维度相同，边界用0填充
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))    # Maxpooling（最大池）方法，指定步长为2，把输出层的两个维度各缩小为一半
        model.add(Conv2D(50, kernel_size=5, padding="same"))    # 第二个卷积层，50个5*5滤波器，注意深层的滤波器数量大于浅层滤波器数量
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 构建全连接层（Flatten层到Relu层）
        model.add(Flatten())        # 展开
        model.add(Dense(500))
        model.add(Activation("relu"))
        # 构建输出层，使用softmax方法
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


# 网络基本参数
NB_EPOCH = 4
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPILT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = IMG_ROWS, IMG_COLS, 1  # 输入矩阵的格式，(28,28,1)

# 混合并划分训练集与测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# k.set_image_dim_ordering("th")      # 选择输入矩阵的格式，th指theano，也就是(1,28,28)，1指维度
# 转换类型与归一化
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# 处理训练集和测试集数据满足输入层需求（60000*[1*28*28]）
print(X_train.shape)    # 读入的数据是(60000, 28, 28)
X_train = X_train[:, :, :, np.newaxis, ]  # 增加一列维度，变成(60000, 28, 28, 1)
X_test = X_test[:, :, :, np.newaxis, ]
print(X_train.shape)    # 输入数据转换成(60000, 28, 28, 1)，符合TensorFlow的要求
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# 处理训练集和测试集输出数据格式，将类向量转换成二值类别矩阵
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# 初始化网络模型
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPILT)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
# 列出全部历史数据
print(history.history.keys())
# 汇总准确率历史数据
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# 汇总损失函数历史数据
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
