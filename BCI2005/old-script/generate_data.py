#!/usr/bin/env python

import numpy as np

def load_data():
    label = list(np.loadtxt('/home/xcat/experiment/BCI2005IIIa/subject1_divide_data/label_subject1.txt'))
#    print type(label[2])
#    print np.isnan(label[2])
    data = [np.loadtxt('/home/xcat/experiment/BCI2005IIIa/subject1_divide_data/data_subject1_trial_{}.txt'.format(i+1)) for i in range(360) if np.isnan(label[i])]
    labels = [label[i]-1 for i in range(360) if not np.isnan(label[i])]
    return np.array(data), np.array(labels)

X_data, y_data = load_data()

np.save('train_data',X_data)
np.save('label_data',y_data)

#print y_data
#print X_data.shape
#print y_data.shape
