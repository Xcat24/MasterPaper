#!/usr/bin/env python
from __future__ import print_function
import numpy as np
np.random.seed(1447)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Convolution2D, Convolution3D, MaxPooling3D, MaxPooling2D, AveragePooling2D, LSTM
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint

batch_size = 2
nb_classes = 2
nb_epoch = 500

# input image dimensions
img_rows, img_cols = 59, 59
# number of convolutional filters to use
nb_filters_1 = 8
nb_filters_2 = 16
# size of pooling area for max pooling
pool_size_1 = (2, 2, 2)
pool_size_2 = (20, 2, 2)
# convolution kernel size
kernel_size_1 = (1, 5, 5)
kernel_size_2 = (1, 3, 3)

X_data = np.load('/media/xcat/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/little_wave/alpha_4s.npy')
y_data = np.load('/media/xcat/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/label.npy')
y_data = y_data/2 + 0.5
#print('x_data:',X_data[0,0:10,:])

X_train = X_data[:160,:,:,:]    #.transpose((0,2,1))
X_test = X_data[160:,:,:,:]     #.transpose((0,2,1))
Y_train = y_data[:160]
Y_test = y_data[160:]
input_shape = (1, 4000, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.reshape((-1,1, 4000, img_rows,img_cols))
X_test = X_test.reshape((-1,1, 4000, img_rows,img_cols))

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print('y_train shape:',Y_train.shape)
print('y_test shape:',Y_test.shape)
model = Sequential()

conv1 = Convolution3D(nb_filters_1, kernel_size_1[0], kernel_size_1[1], kernel_size_1[2], border_mode='same', input_shape=input_shape)
model.add(conv1)
model.add(Activation('relu'))

pool1 = MaxPooling3D(pool_size=pool_size_1)
model.add(pool1)

conv2 = Convolution3D(nb_filters_2, kernel_size_2[0], kernel_size_2[1], kernel_size_2[2], border_mode='same')
model.add(conv2)
model.add(Activation('relu'))

pool2 = MaxPooling3D(pool_size=pool_size_2)
model.add(pool2)

model.add(Permute((2,1,3,4)))
print(model.layers[6].output_shape)
model.add(Reshape((100,16*14*14)))

model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

modelname = '4s_alpha_2BiLSTM_tmr'

plot(model, to_file='/media/xcat/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/model/little_wave/'+modelname+'.png', show_shapes=True)

opt = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='/media/xcat/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/model/little_wave/'+modelname+'.h5',verbose=1,save_best_only=True, monitor='val_acc')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test), callbacks=[checkpointer])

#model.save('/home/xcat/MasterPaper/model/' + modelname + '.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
