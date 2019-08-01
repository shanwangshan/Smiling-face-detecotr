#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:05:59 2018

@author: wangshanshan
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import os, os.path
import cv2
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers.core import Dense, Dropout, Activation, Flatten 
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.models import load_model
dir_of_images= './genki4k/files'
dir_of_labels= './genki4k/labels.txt'

x=[]
y=np.loadtxt(dir_of_labels, usecols=(0)) # only extract the first column 

for i in 1+np.arange(4000):    
    img = cv2.imread(dir_of_images+'/file'+str(i).zfill(4)+'.jpg')
    resized_image = cv2.resize(img, (64,64)) # resize the image to 64*64*3
    resized_image=(resized_image-np.min(resized_image))/np.max(resized_image)
   # resized_image = resized_image.astype("float") / 255.0
    #image = img_to_array(image)
  #  img_=(img-np.min(img))/np.max(img) #normalization

    x.append(resized_image)
x = np.array(x)
y = np.array(y)
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=7)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
model = load_model('smile_dection_model_cnn.h5')
#model.load_weights('./best_weight_smile_cnn.hdf5')
pre=model.predict(X_test)
pre_=pre.round()
acc=accuracy_score(pre_,y_test)
y_test_=np.argmax(y_test,axis=1)
pre__=np.argmax(pre_,axis=1)
matrix=confusion_matrix(y_test_, pre__)
print('accuarcy is testing data is',acc)
print('the confusion matrix is\n', matrix)
# test on the same training data
pre_train=model.predict(X_train)
pre_train=pre_train.round()
acc=accuracy_score(pre_train,y_train)
print('accuarcy of training data is ',acc)