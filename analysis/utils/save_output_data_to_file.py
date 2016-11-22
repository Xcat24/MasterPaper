#!/usr/bin/env python

from __future__ import print_function                                                                                                                                                                   
import numpy as np                                                                                                                                                                                      


from keras.models import Sequential, load_model 
from keras import backend as K                                                                                                                                                                          

# input image dimensions                                                                                                                              
img_rows, img_cols = 4000, 59 
X_data = np.load('/home/xcat/experiment/BCI2008/ds1a/train.npy')
y_data = np.load('/home/xcat/experiment/BCI2008/ds1a/label.npy')
#scale data 
temp_max = np.fabs(X_data).max()
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

np.save('/home/xcat/experiment/BCI2008/ds1a/input_data', X_train)
model = load_model("/home/xcat/MasterPaper/model/model_2.3_RMSprop_lr0.001.h5")
#print(model.get_config())
#print(model.layers[0].get_config())

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

if __name__ == '__main__':
    save_conv1_out()
    save_pool1_out()
    save_conv2_out()
    save_pool2_out()


