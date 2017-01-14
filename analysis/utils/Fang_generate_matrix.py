#!/usr/bin/env python

import numpy as np

trial_num1 = 45
trial_num2 = 40
pole_num = 64
window_num = 5
save_name = '/media/xcat/9A8821088820E48B/Fang/train'
load_path1 = '/media/xcat/9A8821088820E48B/Fang/brainnetfeature9/'
load_path2 = '/media/xcat/9A8821088820E48B/Fang/brainnetfeature10/'

data1 = np.zeros((trial_num1, window_num, pole_num, pole_num))
for trial in range(trial_num1):
    trial_data = np.loadtxt(load_path1 + '2-Chente-yulan-b-a-9' + str(trial+1) + '-thetaalpha.dat.txt')
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
for trial in range(trial_num2):
    trial_data = np.loadtxt(load_path2 + '2-Chente-yulan-b-a-10' + str(trial+1) + '-thetaalpha.dat.txt')
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

