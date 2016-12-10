#!/usr/bin/env python

from __future__ import print_function 
import numpy as np 

from keras.models import Sequential, load_model 
from keras import backend as K 

ModelName = '/home/xcat/experiment/BCI2008/ds1a/model/2lstm_false_RMSprop_lr0.001/Li_model_RMSprop_lr0.001.h5'
OutputPath = '/home/xcat/experiment/BCI2008/ds1a/model/2lstm_false_RMSprop_lr0.001/'
pos_index = 0
neg_index = 3
img_rows, img_cols = 4000, 59 
X_data = np.load('/home/xcat/experiment/BCI2008/ds1a/train.npy')
y_data = np.load('/home/xcat/experiment/BCI2008/ds1a/label.npy')
#scale data 
temp_max = np.fabs(X_data).max()
X_data = X_data/temp_max
y_data = y_data/2 + 0.5

pos_sample = X_data[pos_index]
neg_sample = X_data[neg_index]
pos_sample = pos_sample.reshape((-1,1,img_rows,img_cols))
neg_sample = neg_sample.reshape((-1,1,img_rows,img_cols))

model = load_model(ModelName)
#print(model.get_config())
#print(model.layers[0].get_config())
get_conv1_output = K.function([model.layers[0].input], [model.layers[1].output])
get_pool1_output = K.function([model.layers[0].input], [model.layers[2].output])
get_conv2_output = K.function([model.layers[0].input], [model.layers[4].output])
get_pool2_output = K.function([model.layers[0].input], [model.layers[7].output])

conv1_output_pos = get_conv1_output([pos_sample])[0]  #numpy.ndarray
pool1_output_pos = get_pool1_output([pos_sample])[0]  #numpy.ndarray
conv2_output_pos = get_conv2_output([pos_sample])[0]  #numpy.ndarray
pool2_output_pos = get_pool2_output([pos_sample])[0]  #numpy.ndarray
conv1_output_neg = get_conv1_output([neg_sample])[0]  #numpy.ndarray
pool1_output_neg = get_pool1_output([neg_sample])[0]  #numpy.ndarray
conv2_output_neg = get_conv2_output([neg_sample])[0]  #numpy.ndarray
pool2_output_neg = get_pool2_output([neg_sample])[0]  #numpy.ndarray

np.save(OutputPath+model.layers[1].name, conv1_output_pos)
np.save(OutputPath+model.layers[2].name, pool1_output_pos)
np.save(OutputPath+model.layers[4].name, conv2_output_pos)
np.save(OutputPath+model.layers[7].name, pool2_output_pos)
np.save(OutputPath+model.layers[1].name, conv1_output_neg)
np.save(OutputPath+model.layers[2].name, pool1_output_neg)
np.save(OutputPath+model.layers[4].name, conv2_output_neg)
np.save(OutputPath+model.layers[7].name, pool2_output_neg)


def save_conv1_out():
        #define function to get output of some layer
        get_conv1_output = K.function([model.layers[0].input], [model.layers[1].output])
        #print(type(get_conv1_output([X_train[0:0,:,:,:]])[0]))  #numpy.ndarray
        conv1_output = get_conv1_output([X_train[:,:,:,:]])[0]  #numpy.ndarray
        np.save('/home/xcat/experiment/BCI2008/ds1a/model2.3_'+model.layers[1].name, conv1_output)
        return

def save_pool1_out():
    get_pool1_output = K.function([model.layers[0].input], [model.layers[2].output])
    pool1_output = np.zeros((160,40,2000,59))
    for i in range(160):
        pool1_output[i] = get_pool1_output([X_train[i:i+1,:,:,:]])[0]  #numpy.ndarray
        np.save('/home/xcat/experiment/BCI2008/ds1a/model2.3_'+model.layers[2].name, pool1_output)
        return

def save_conv2_out():
    get_conv2_output = K.function([model.layers[0].input], [model.layers[4].output])
    conv2_output = np.zeros((160,40,2000,59))
    for i in range(160):
        conv2_output[i] = get_conv2_output([X_train[i:i+1,:,:,:]])[0]  #numpy.ndarray
        np.save('/home/xcat/experiment/BCI2008/ds1a/model2.3_'+model.layers[4].name, conv2_output)
        return

def save_pool2_out():
    get_pool2_output = K.function([model.layers[0].input], [model.layers[7].output])
    pool2_output = np.zeros((160,2000,59))
    for i in range(160):
        pool2_output[i] = get_pool2_output([X_train[i:i+1,:,:]])[0]  #numpy.ndarray
        np.save('/home/xcat/experiment/BCI2008/ds1a/model2.3_'+model.layers[7].name, pool2_output)
        return

#if __name__ == '__main__':
#    save_conv1_out()
#    save_pool1_out()
#    save_conv2_out()
#    save_pool2_out()


