from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
NUM_TO_AUGMENT = 5


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


# 图片基本信息
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
# 网络基本参数
NB_EPOCH = 30
BATCH_SIZE = 128
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = RMSprop()
# 读入数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape)
# 对输出结果y进行分类转换
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# 归一化处理
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# 构建模型
model = ComplexNetWork.build(INPUT_SHAPE, NB_CLASSES)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# 增加数据集， rotation_range 表示翻转角度(1~180)， width_shift和height_shift表示对图片做随机水平或垂直变化的范围。
# zoom_range表示随机缩放图片的变化值，horizontal_flip是对选中的一半图片进行随机的水平翻转， fill_mode是图片翻转或交换后用来填充新像素的策略
print("Augmenting training set images...")
# datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0,
#                              horizontal_flip=True, fill_mode='nearest')
datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

# xtas, ytas = [], []
# for i in range(X_train.shape[0]):
#     num_aug = 0
#     x = X_train[i]
#     x = x.reshape((1,) + x.shape)   # (1, 3, 32, 32)
#     for x_aug in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cifar', save_format='jpeg'):
#         if num_aug >= NUM_TO_AUGMENT:
#             break
#         xtas.append(x_aug[0])
#         num_aug += 1

# 匹配数据
datagen.fit(X_train)
print(X_train.shape, y_train.shape)

# 训练模型
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=X_train.shape[0],
                             epochs=NB_EPOCH, verbose=VERBOSE)  # 一边生成增强数据一边训练
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
