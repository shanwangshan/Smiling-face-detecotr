#!/uisr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:18:21 2018

@author: wang9
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
import sklearn
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

num_featmaps = 32 
num_classes = 2
num_epochs = 200
w, h = 5,5
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

#define the model
model = Sequential()
model.add(Conv2D(num_featmaps, (w, h), input_shape=(64, 64, 3),activation = 'relu'))
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu')) 
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu')) 
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
checkpoint = ModelCheckpoint('./best_weight_smile_cnn.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss',  patience=30, verbose=1, mode='auto')
callbacks_list = [checkpoint, early_stopping]
# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics = ['accuracy'])
history=model.fit(X_train, y_train, epochs = num_epochs, validation_data = (X_test, y_test),callbacks=callbacks_list)
model.save('./smile_dection_model_cnn.h5')
model.save_weights('./ending_weight_smile_dection_model_cnn.h5')

#plot the training figure
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['acc'], 'ro-', label = "Train Accuracy")
ax[0].plot(history.history['val_acc'], 'go-', label = "Test Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy / %")
ax[0].legend(loc = "best")
ax[0].grid('on')

ax[1].plot(history.history['loss'], 'ro-', label = "Train Loss")
ax[1].plot(history.history['val_loss'], 'go-', label = "Test Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend(loc = "best")
ax[1].grid('on')

plt.tight_layout()
plt.savefig("Accuracy_cnn.pdf", bbox_inches = "tight")
