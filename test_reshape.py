#!/usr/bin/env python
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

nb_epoch = 1

X_data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).reshape((1,3,4))
print(X_data)
model = Sequential()

model.add(Reshape((4,3),input_shape=(3,4)))
model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.save('test.h5')

#model.predict(X_data)
print(model.predict(X_data))

