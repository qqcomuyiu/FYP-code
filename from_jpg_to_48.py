# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
        #print(array_of_img)

path='facial//test_image'
read_directory(path)

#for img in files:
   # image=load_img(img)
#image=load_img('E:\\C\Cadence\\SPB_Data\\PrivateTest\\test\\Angry_00005.jpg')
