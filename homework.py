import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt


input_shape = 2
NB_EPOCH = 4000
BATCH_SIZE = 128
NB_CLASSES = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = Adam()

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
print(X_train)
model = Sequential()
model.add(Dense(64, input_shape=(input_shape,)))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
hist = model.fit(X_train, y_train, verbose=VERBOSE, epochs=NB_EPOCH, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
plt.scatter(range(len(hist.history['loss'])), hist.history['loss'])
loss = model.evaluate(X_test, y_test)
print(loss)
plt.show()


