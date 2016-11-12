#!/usr/bin/env python

from scipy.misc import imsave
import numpy as np

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    #clip to [0,1]
    x += 0.5
    x = np.clip(x,0,1)
    #convert to RGB array
    x *= 255
#    x = x.transpose((1,2,0))
    x = np.clip(x,0,255).astype('uint8')

    return x

def save_img(x, path='/home/xcat/experiment/analysis/'):
    try:
        img = deprocess_image(x)
        imsave(path,img)
    except:
        print('save data to image ERROR!')


if __name__ == '__main__':
    img = np.random.random((1000,59,3))
    img = deprocess_image(img)
    imsave('test.png',img)

