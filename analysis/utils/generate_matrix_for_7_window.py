#!/usr/bin/env python

import numpy as np

trial_num = 200
pole_num = 59
window_num = 7
#window 0: 0.0~1.0s
#window 1: 0.5~1.5s
#window 2: 1.0~2.0s
#window 3: 1.5~2.5s
#window 4: 2.0~3.0s
#window 5: 2.5~3.5s
#window 6: 3.0~4.0s
save_name = '/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/7-window-4000-data/'
load_path = '/media/kong/9A8821088820E48B/BCI2008/processAdjMatrix_output/1711_7/'

data = np.zeros((trial_num, window_num, pole_num, pole_num))

for trial in range(trial_num):
    trial_data = np.loadtxt(load_path + str(trial+1) + '-all.dat.txt')
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

