#!/usr/bin/env python
'''add data scale module
data map to [-1,1]
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

batch_size = 128
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 400, 59
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

def load_data():
    data = [np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]
    label = np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/labels_BCI2008_ds1a.txt')
    return np.array(data),label

X_data, y_data = load_data()
#scale data
temp_max = np.fabs(X_data).max()
print('max:', temp_max)
X_data = X_data/temp_max
print('x_data:',X_data[0,0:10,:])

X_train = X_data[:160,2000:2400,:]    #.transpose((0,2,1))
X_test = X_data[160:,2000:2400,:]     #.transpose((0,2,1))
Y_train = y_data[:160]
Y_test = y_data[160:]
input_shape = (1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.reshape((-1,1,img_rows,img_cols))
X_test = X_test.reshape((-1,1,img_rows,img_cols))

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print('y_train shape:',Y_train.shape)
print('y_test shape:',Y_test.shape)

model = Sequential()

conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape)
model.add(conv1)
print('conv1\'s output shape:',conv1.output_shape)

model.add(Activation('relu'))

conv2 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same')
model.add(conv2)
print('conv2\'s output shape:',conv2.output_shape)

model.add(Activation('relu'))

pool1 = MaxPooling2D(pool_size=pool_size)
model.add(pool1)
print('pool1\'s output shape:', pool1.output_shape)

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
