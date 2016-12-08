#!/usr/bin/env python

from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt

def repeat_pixel(data, x=1):
    #use x pixel to represent 1 column in a 2D-matrix
    temp = np.zeros((data.shape[0], x*data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(x):
            temp[:,i*x+j] = data[:,i]
    return temp

def deprocess_image(x):
#    x -= x.mean()
#    x /= (x.std() + 1e-5)
#    x *= 0.1
    #clip to [0,1]
    print(x.max())
    x /= x.max()
    x /= 2
    x += 0.5
    x = np.clip(x,0,1)
    #convert to RGB array
    x *= 255
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x[0]
    x = np.clip(x,0,255).astype('uint8')
    return x

def deprocess_to_wave(x):
    if x.ndim != 2:
        raise
    output_matrix = np.zeros((x.shape[0],30*x.shape[1]))
    print(x.max())
    x /= (x.max()*0.5)
    x /= 2
    x += 0.5
    x = np.clip(x,0,1)
    print(x.max())
    x *= 30
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if 30*j+round(x[i,j]) < 30*x.shape[1]:
                output_matrix[i,30*j+round(x[i,j])] = 255
            else:
                output_matrix[i,30*x.shape[1]-1] = 255
    output_matrix = output_matrix.astype('uint8')
    return output_matrix

def save_img(x, path='/home/xcat/experiment/BCI2008/ds1a/picture/'):
#    try:
        img = deprocess_image(x)
        print(img.shape)
        imsave(path,img)
#    except:
#        print('save data to image ERROR!')

def input_data_to_img():
    data = np.load('/home/xcat/experiment/BCI2008/ds1a/train.npy')
    label = np.load('/home/xcat/experiment/BCI2008/ds1a/label.npy')
#    print(label.shape)  #(200,)
#    print(label[0:5])   #(1,1,-1,...)
    for i in range(len(data)):
        img = deprocess_image(data[i])
        if img.ndim != 2:
            raise
        img = repeat_pixel(img, 15)
        if label[i] == 1:
            imsave('/home/xcat/experiment/BCI2008/ds1a/picture/input_data/train_tiral{}_+.png'.format(i), img)
        elif label[i] == -1:
            imsave('/home/xcat/experiment/BCI2008/ds1a/picture/input_data/train_tiral{}_-.png'.format(i), img)
    return

if __name__ == '__main__':
#test repeat_pixel #DONE!
#    a = np.array([[1,2],[3,4]])
#    print(a)
#    print(repeat_pixel(a,5))

#test plot input data in wave format
    data = np.load('/home/xcat/experiment/BCI2008/ds1a/train.npy')
    img = deprocess_to_wave(data[0])
    imsave('/home/xcat/experiment/BCI2008/ds1a/picture/input_data/wave_form/trial_1-orgin.png',img)
#test save some output npy data to image #DONE
#    data = np.load('/home/xcat/experiment/BCI2008/ds1a/model/2lstm_false_RMSprop_lr0.001/activation_1.npy')
#    img = deprocess_image(data[0])
#    imsave('/home/xcat/experiment/BCI2008/ds1a/picture/2lstm_false_RMSprop_lr0.001/activation1.png', img)  
#    data = np.load('/home/xcat/experiment/BCI2008/ds1a/model/2lstm_false_RMSprop_lr0.001/activation_2.npy')
#    img = deprocess_image(data[0])
#    imsave('/home/xcat/experiment/BCI2008/ds1a/picture/2lstm_false_RMSprop_lr0.001/activation2.png', img)  
#    data = np.load('/home/xcat/experiment/BCI2008/ds1a/model/2lstm_false_RMSprop_lr0.001/averagepooling2d_1.npy')
#    img = deprocess_image(data[0])
#    imsave('/home/xcat/experiment/BCI2008/ds1a/picture/2lstm_false_RMSprop_lr0.001/averagepooling2d_1.png', img)  
#    data = np.load('/home/xcat/experiment/BCI2008/ds1a/model/2lstm_false_RMSprop_lr0.001/reshape_1.npy')
#    img = deprocess_image(data[0])
#    imsave('/home/xcat/experiment/BCI2008/ds1a/picture/2lstm_false_RMSprop_lr0.001/reshape_1.png', img)  
