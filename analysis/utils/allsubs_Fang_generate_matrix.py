#!/usr/bin/env python

import numpy as np
import os

trial_num1 = 549
trial_num2 = 530
pole_num = 64
window_num = 5
save_name = '/media/xcat/9A8821088820E48B/Fang/allsubs_train'
load_path1 = '/media/xcat/9A8821088820E48B/Fang/brainnetfeature9/'
load_path2 = '/media/xcat/9A8821088820E48B/Fang/brainnetfeature10/'

data1 = np.zeros((trial_num1, window_num, pole_num, pole_num))
datafiles = os.listdir(load_path1)
for trial in range(len(datafiles)):
    trial_data = np.loadtxt(load_path1 + datafiles[trial])
    for window in range(trial_data.shape[1]):
        for pole_row in range(pole_num):
            for pole_col in range(pole_num):
                if pole_row < pole_col : #up part
                    if pole_row == 0:
                        data1[trial,window,pole_row,pole_col] = trial_data[pole_col - pole_row - 1, window]
                    else:
                        data1[trial,window,pole_row,pole_col] = trial_data[(pole_num-1 + pole_num-pole_row)*pole_row/2 + pole_col - pole_row - 1, window]
                elif pole_row > pole_col : #down part
                    data1[trial,window,pole_row,pole_col] = data1[trial,window,pole_row,pole_col]
                elif pole_row == pole_col :
                    data1[trial,window,pole_row,pole_col] = 1.0

data2 = np.zeros((trial_num2, window_num, pole_num, pole_num))
datafiles = os.listdir(load_path2)
for trial in range(len(datafiles)):
    trial_data = np.loadtxt(load_path2 + datafiles[trial])
    for window in range(trial_data.shape[1]):
        for pole_row in range(pole_num):
            for pole_col in range(pole_num):
                if pole_row < pole_col : #up part
                    if pole_row == 0:
                        data2[trial,window,pole_row,pole_col] = trial_data[pole_col - pole_row - 1, window]
                    else:
                        data2[trial,window,pole_row,pole_col] = trial_data[(pole_num-1 + pole_num-pole_row)*pole_row/2 + pole_col - pole_row - 1, window]
                elif pole_row > pole_col : #down part
                    data2[trial,window,pole_row,pole_col] = data2[trial,window,pole_row,pole_col]
                elif pole_row == pole_col :
                    data2[trial,window,pole_row,pole_col] = 1.0

np.save(save_name,np.vstack((data1,data2)))
label = np.zeros(1079)
label[549:] = 1
np.save('/media/xcat/9A8821088820E48B/Fang/allsubs_label',label)
