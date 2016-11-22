#!/usr/bin/env python

from scipy.misc import imsave
import numpy as np

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
    x /= x.max()
    x /= 2
    x += 0.5
    x = np.clip(x,0,1)
    #convert to RGB array
    x *= 255
    if x.ndim == 3:
        x = x.transpose((1,2,0))
    elif x.ndim ==4:
        x = x[0]
        x = x.transpose((1,2,0))
    x = np.clip(x,0,255).astype('uint8')

    return x

def save_img(x, path='/home/xcat/experiment/analysis/'):
#    try:
        img = deprocess_image(x)
        print(img.shape)
        imsave(path,img)
#    except:
#        print('save data to image ERROR!')


if __name__ == '__main__':
'''
-------------------------------------------------------------------------
    visualize the original input data
-------------------------------------------------------------------------
'''
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

#test repeat_pixel

#    a = np.array([[1,2],[3,4]])
#    print(a)
#    print(repeat_pixel(a,5))

