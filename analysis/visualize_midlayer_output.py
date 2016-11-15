#!/usr/bin/env python

from __future__ import print_function                                                                                                                                                                   
import numpy as np                                                                                                                                                                                      


from keras.models import Sequential, load_model 
from keras import backend as K                                                                                                                                                                          
from keras.utils.visualize_util import plot
from utils.visualize_data import save_img

def load_data():                                                                                                                                                                                        
    data = [np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]                                                   
    label = np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/labels_BCI2008_ds1a.txt')                                                                                           
    return np.array(data),label                                                                                                                                                                         

# input image dimensions                                                                                                                                                                                
img_rows, img_cols = 4000, 59                                                                                                                                                                                                     
X_data, y_data = load_data()                                                                                                                                                                            
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


model = load_model("/home/xcat/MasterPaper/model/model_2.2_RMSprop_lr0.001.h5")
#print(model.get_config())
#print(model.layers[0].get_config())

#define function to get output of some layer                                                                                                                                                            
get_conv1_output = K.function([model.layers[0].input], [model.layers[1].output])
get_pool1_output = K.function([model.layers[0].input], [model.layers[2].output])
get_conv2_output = K.function([model.layers[0].input], [model.layers[4].output])
get_pool2_output = K.function([model.layers[0].input], [model.layers[7].output])

#print(type(get_conv1_output([X_train[0:0,:,:,:]])[0]))  #numpy.ndarray
conv1_output = get_conv1_output([X_train[0:1,:,:,:]])[0]  #numpy.ndarray
pool1_output = get_pool1_output([X_train[0:1,:,:,:]])[0]  #numpy.ndarray
conv2_output = get_conv2_output([X_train[0:1,:,:,:]])[0]  #numpy.ndarray
pool2_output = get_pool2_output([X_train[0:1,:,:,:]])[0]  #numpy.ndarray

np.save('model2.2_'+model.layers[1].name, conv1_output)
np.save('model2.2_'+model.layers[2].name, pool1_output)
np.save('model2.2_'+model.layers[4].name, conv2_output)
np.save('model2.2_'+model.layers[7].name, pool2_output)

save_img(conv1_output[0,0,:,:], 'model2.2_'+model.layers[1].name+'.png')
save_img(pool1_output[0,0,:,:], 'model2.2_'+model.layers[2].name+'.png')
save_img(conv2_output[0,0,:,:], 'model2.2_'+model.layers[4].name+'.png')
save_img(pool2_output[0,:,:], 'model2.2_'+model.layers[7].name+'.png')

