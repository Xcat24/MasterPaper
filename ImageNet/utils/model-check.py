#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from keras.models import Sequential, load_model
from keras import backend as K

if __name__ == '__main__':
    model = load_model('/home/xcat/experiment/ImageNet/model/image-methodh5')
    test = np.load('/media/kong/9A8821088820E48B/ImageNet_data/test_data.npy')
    result = model.predict_classes(test)
    print(result)
