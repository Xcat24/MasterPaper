#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os

ImageNet_Path =  '/media/kong/9A8821088820E48B/ImageNet/'
dir_list = ['n02091467/','n02132136/']
TrainDataNum = 1000
TestDataNum = 300
img_height = 224
img_width = 224

def img_to_npy(save_path, path=ImageNet_Path):
    train_data = np.zeros((TrainDataNum*len(dir_list),3,img_height,img_width))
    test_data = np.zeros((TestDataNum*len(dir_list),3,img_height,img_width))
    train_label = np.zeros(TrainDataNum*len(dir_list))
    test_label = np.zeros(TestDataNum*len(dir_list))
    for i in range(len(dir_list)):
        cnt = 0
        for img_name in os.listdir(path + dir_list[i]):
            if cnt < TrainDataNum:
                img = load_img(path+dir_list[i]+img_name, target_size=(img_height, img_width))
                img_npy = img_to_array(img)
                train_data[i*TrainDataNum + cnt] = img_npy
                train_label[i*TrainDataNum + cnt] = i
            elif cnt < TrainDataNum + TestDataNum:
                img = load_img(path + dir_list[i] + img_name, target_size=(img_height, img_width))
                img_npy = img_to_array(img)
                test_data[i*TestDataNum + cnt - TrainDataNum] = img_npy 
                test_label[i*TestDataNum + cnt - TrainDataNum] = i
            else:
                break 
            cnt += 1
            
    np.save(save_path+'train_data', train_data)
    np.save(save_path+'test_data', test_data)
    np.save(save_path+'train_label', train_label)
    np.save(save_path+'test_label', test_label)
    return


if __name__ == '__main__':
    img_to_npy('/media/kong/9A8821088820E48B/ImageNet_data/')    
