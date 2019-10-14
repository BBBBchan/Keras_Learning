from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
np.random.seed(1671)

# 基本参数设置
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10     # 分类数量，对应10个数字
# OPTIMIZER = SGD()   # SGD优化器，随机梯度下降(Stochastic Gradient Descent)
OPTIMIZER = Adagrad()  # Adagrad优化器
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # 训练集中选出用作验证集的比例
DROPOUT = 0.3   # 随机丢弃结果比例

# 数据导入和预处理
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train 是60000行28*28的数据，现在将其变形为6000*784的矩阵
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 归一化处理
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 对结果向量进行转换
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 网络设计
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))   # 第一次加入网络，指定输入层大小和神经元个数
model.add(Activation('relu'))       # 指定激活函数
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))          # 再增加一层网络，指定神经元个数
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))        # 加入输出层网络
model.add(Activation('softmax'))    # 选择激活函数
model.summary()
# 编译模型，指定损失函数，优化器和评测指标
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# 训练模型，给定训练集的输入向量和输出向量，batch_size指更新一次权重前要考虑的实例数， epochs指训练轮数
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
