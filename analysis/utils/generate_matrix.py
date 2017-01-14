#!/usr/bin/env python

import numpy as np

trial_num = 200
pole_num = 59
window_len = 4000
save_name = '/media/xcat/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/little_wave/alpha_4s'
load_path = '/media/xcat/9A8821088820E48B/BCI2008/4000_feature/'

data = np.zeros((trial_num, window_len, pole_num, pole_num))

for trial in range(trial_num):
    trial_data = np.loadtxt(load_path + str(trial+1) + '/alpha.dat')
    for window in range(trial_data.shape[1]):
        for pole_row in range(pole_num):
            for pole_col in range(pole_num):
                if pole_row < pole_col : #up part
                    if pole_row == 0:
                        data[trial,window,pole_row,pole_col] = trial_data[pole_col - pole_row - 1, window]
                    else:
                        data[trial,window,pole_row,pole_col] = trial_data[(pole_num-1 + pole_num-pole_row)*pole_row/2 + pole_col - pole_row - 1, window]
                elif pole_row > pole_col : #down part
                    data[trial,window,pole_row,pole_col] = data[trial,window,pole_row,pole_col]
                elif pole_row == pole_col :
                    data[trial,window,pole_row,pole_col] = 1.0

np.save(save_name,data)

