#!/usr/bin/env python
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, LSTM
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint

batch_size = 10
nb_classes = 5
nb_epoch = 500

# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (3,224,224)
# number of convolutional filters to use
nb_filters = 40
# size of pooling area for max pooling
pool_size_1 = (2, 2)
pool_size_2 = (2, 2)
# convolution kernel size
kernel_size_1 = (7, 7)
kernel_size_2 = (5, 5)

X_train = np.load('/media/kong/9A8821088820E48B/ImageNet_data/5-class-train_data.npy')
Y_train = np.load('/media/kong/9A8821088820E48B/ImageNet_data/5-class-train_label.npy')
X_test = np.load('/media/kong/9A8821088820E48B/ImageNet_data/5-class-test_data.npy')
Y_test = np.load('/media/kong/9A8821088820E48B/ImageNet_data/5-class-test_label.npy')
#scale data
X_train /= 255.0
X_test /= 255.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print('y_train shape:',Y_train.shape)
print('y_test shape:',Y_test.shape)
model = Sequential()

conv1 = Convolution2D(nb_filters, kernel_size_1[0], kernel_size_1[1], border_mode='same', input_shape=input_shape)
model.add(conv1)
model.add(Activation('relu'))

pool1 = MaxPooling2D(pool_size=pool_size_1)
model.add(pool1)

conv2 = Convolution2D(nb_filters, kernel_size_2[0], kernel_size_2[1], border_mode='same')
model.add(conv2)
model.add(Activation('relu'))

pool2 = MaxPooling2D(pool_size=pool_size_2)
model.add(pool2)

model.add(Permute((2,1,3)))
model.add(Reshape((56,40*56)))

model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

modelname = '5-class-image-method'

plot(model, to_file='/home/xcat/experiment/ImageNet/model/'+modelname+'.png', show_shapes=True)

opt = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='/home/xcat/experiment/ImageNet/model/'+modelname+'h5',verbose=1,save_best_only=True, monitor='val_acc')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test), callbacks=[checkpointer])

#model.save('/home/xcat/MasterPaper/model/' + modelname + '.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
