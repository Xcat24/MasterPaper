#!/usr/bin/env python

import numpy as np

def read_1000Hz_data_to_npy():                                                                                                                                                                                        
    data = [np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]                                                   
    label = np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/labels_BCI2008_ds1a.txt')                                                                                           
    np.save('/home/xcat/experiment/BCI2008/ds1a/train',np.array(data))
    np.save('/home/xcat/experiment/BCI2008/ds1a/label',label)
    return

if __name__ == '__main__':
    read_1000Hz_data_to_npy()

