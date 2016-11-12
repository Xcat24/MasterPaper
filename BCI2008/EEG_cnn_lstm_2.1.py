#!/usr/bin/env python
'''change LSTM layer hidden nodes from 2 to 128
delete dropout layers
change pool2 layer to cross channel
'''

from __future__ import print_function
import numpy as np
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.visualize_util import plot

batch_size = 10
nb_classes = 2
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 4000, 59
# number of convolutional filters to use
nb_filters = 40
# size of pooling area for max pooling
pool_size_1 = (2, 1)
pool_size_2 = (40,1)
# convolution kernel size
kernel_size_1 = (5, 1)
kernel_size_2 = (500,1)

def load_data():
    data = [np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]
    label = np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/labels_BCI2008_ds1a.txt')
    return np.array(data),label

X_data, y_data = load_data()
#scale data
temp_max = np.fabs(X_data).max()
print('max:', temp_max)
X_data = X_data/temp_max
y_data = y_data/2 + 0.5
#print('x_data:',X_data[0,0:10,:])

X_train = X_data[:160,0:4000,:]    #.transpose((0,2,1))
X_test = X_data[160:,0:4000,:]     #.transpose((0,2,1))
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
print(Y_train[0:15,:])
model = Sequential()

conv1 = Convolution2D(nb_filters, kernel_size_1[0], kernel_size_1[1], border_mode='same', input_shape=input_shape)
model.add(conv1)
print('conv1\'s output shape:',conv1.output_shape)

model.add(Activation('relu'))

pool1 = MaxPooling2D(pool_size=pool_size_1)
model.add(pool1)
print('pool1\'s output shape:', pool1.output_shape)

conv2 = Convolution2D(nb_filters, kernel_size_2[0], kernel_size_2[1], border_mode='same')
model.add(conv2)
print('conv2\'s output shape:',conv2.output_shape)

model.add(Activation('relu'))
model.add(Permute((2,1,3)))

pool2 = MaxPooling2D(pool_size=pool_size_2)
model.add(pool2)
print('pool2\'s output shape:', pool2.output_shape)

#model.add(Dropout(0.25))

reshape = Reshape((2000,59))
model.add(reshape)
print('reshape\'s output shape:',reshape.output_shape)
model.add(Permute((2,1)))

model.add(LSTM(128))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()
plot(model, to_file='model.png', show_shapes=True)

opt = Adam(lr=0.002)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
