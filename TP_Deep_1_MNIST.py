#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:11:51 2018

@author: stephane
# http://yann.lecun.com/exdb/mnist/
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import time

#%%
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

import numpy as np
print(np.shape(x_train))


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt
gen_image(x_train[0]).show()
print(np.nonzero(y_train[0])[0][0])

#%%

import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

#%%

# 2-layer NN, 300 hidden units, mean square error 	none 	4.7 	LeCun et al. 1998
model = Sequential()
model.add(Dense(units=300, activation='tanh', input_dim=784))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False), 
 #             optimizer=keras.optimizers.Adadelta(), 
              metrics=['accuracy'])

model.summary()
              
t0=time.time()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
print('')
print('total computing time:'+str(time.time()-t0))

model.evaluate(x_test, y_test)  # err = 6.12 -> 2 % with keras.optimizers.Adadelta
# https://arxiv.org/abs/1212.5701
#  ADADELTA: An Adaptive Learning Rate Method
#%%

# 3-layer NN, 500+300 HU, softmax, cross entropy, weight decay 	 	1.53 	Hinton, unpublished, 2005
model1 = Sequential()
model1.add(Dense(units=500, activation='relu', input_dim=784))
#model.add(Dense(units=300, activation='relu'))
# model.add(Dropout(0.2))
model1.add(Dense(units=300, activation='tanh'))
model1.add(Dense(units=10, activation='softmax'))

model1.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(), 
              metrics=['accuracy'])

model1.summary()
              
t0=time.time()
model1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
print('')
print('total computing time:'+str(time.time()-t0))

score = model1.evaluate(x_test, y_test)  # err = 1.46 %
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
# Convolutional net LeNet-5, [no distortions] 	 	0.95 	LeCun et al. 1998
from keras.layers import Conv2D, MaxPooling2D

# input image shape
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#%%
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(5, 5),
           activation='relu',
          input_shape=input_shape))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))
model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
		
model2.summary()

#%%
        
t0=time.time()
model2.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
print('')
print('total computing time:'+str(time.time()-t0))

score = model2.evaluate(x_test, y_test)  # err = 0.83 %  en 15 minutes
print('Test loss:', score[0])
print('Test accuracy:', score[1])
     
        
        
        
        