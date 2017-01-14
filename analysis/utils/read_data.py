#!/usr/bin/env python

import numpy as np

def read_Fang_data_to_npy():
    data1 = [np.loadtxt('/media/xcat/9A8821088820E48B/Fang/brainnetfeature9/2-Chente-yulan-b-a-9{}-thetaalpha.dat.txt'.format(i+1)) for i in range(45)]
    data2 = [np.loadtxt('/media/xcat/9A8821088820E48B/Fang/brainnetfeature10/2-Chente-yulan-b-a-10{}-thetaalpha.dat.txt'.format(i+1)) for i in range(40)]
    label = np.zeros(85)
    label[45:] = 1
    data = np.vstack((data1,data2))
    np.save('/media/xcat/9A8821088820E48B/Fang/train',np.array(data))
    np.save('/media/xcat/9A8821088820E48B/Fang/label',label)
    return

def read_1000Hz_data_to_npy():
    data = [np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]
    label = np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCICIV_calib_ds1a_1000Hz/labels_BCI2008_ds1a.txt')
    np.save('/home/xcat/experiment/BCI2008/ds1a/train',np.array(data))
    np.save('/home/xcat/experiment/BCI2008/ds1a/label',label)
    return

def read_8s_data_to_npy():
    data = [np.loadtxt('/media/kong/9A8821088820E48B/Xcat/ds1a_divided_data_8s/data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]
    label = np.loadtxt('/media/kong/9A8821088820E48B/Xcat/ds1a_divided_data_8s/labels_BCI2008_ds1a.txt')
    np.save('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/8s_data/train',np.array(data))
    np.save('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2008/ds1a/8s_data/label',label)
    return

def read_BCI2005IV_to_npy():
    data = [np.loadtxt('/media/kong/9A8821088820E48B/BCI2005_IVa_ASCII/subject1/trial{}.txt'.format(i+1)) for i in range(280)]
    label = np.loadtxt('/media/kong/9A8821088820E48B/BCI2005_IVa_ASCII/subject1/label.txt')
    np.save('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2005IVa/data/subject1/train',np.array(data))
    np.save('/media/kong/9A8821088820E48B/Xcat/experiment/BCI2005IVa/data/subject1/label',label)
    return

def read_filtered_data_to_npy():
    data = [np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCI2008-ds1a-8-30filter-1000Hz/filter_data_BCI2008_ds1a_trial_{}.txt'.format(i+1)) for i in range(200)]
    label = np.loadtxt('/home/xcat/experiment/MrsFang_Task/BCI2008-ds1a-8-30filter-1000Hz/labels_BCI2008_ds1a_filtered.txt')
    np.save('/home/xcat/experiment/BCI2008/ds1a/filtered-train',np.array(data))
    np.save('/home/xcat/experiment/BCI2008/ds1a/filtered-label',label)
    return

if __name__ == '__main__':
#    read_1000Hz_data_to_npy()
#    read_filtered_data_to_npy()
#	read_8s_data_to_npy()
#	read_BCI2005IV_to_npy()
	read_Fang_data_to_npy()

