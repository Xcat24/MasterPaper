#!/usr/bin/env python

import numpy as np
pos = [1,5,1,15,4,4,4,6,4,8,4,10,4,12,4,14,4,16,6,4,6,6,6,8,6,10,6,12,6,14,6,16,8,3,8,5,8,7,8,9,
       8,11,8,13,8,15,8,17,10,2,10,4,10,6,10,8,10,10,10,12,10,14,10,16,10,18,12,3,12,5,12,7,12,9,
       12,11,12,13,12,15,12,17,14,4,14,6,14,8,14,10,14,12,14,14,14,16,16,4,16,6,16,8,16,10,16,12,
       16,14,16,16,18,9,18,11,20,8,20,12]

#data = np.load('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/train.npy')
data = np.load('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/filtered-train.npy')
NewData = np.zeros((200,4000,21,19))

for i in range(59):
    NewData[:,:,pos[2*i]-1,pos[2*i+1]-1] = data[:,:,i]

#np.save('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/arange_pole/train',NewData)
np.save('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/arange_pole/filtered-train',NewData)
