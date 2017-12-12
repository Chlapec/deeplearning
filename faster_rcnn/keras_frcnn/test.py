#coding=utf-8
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt

from keras.datasets import mnist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )


# 获取
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 合并数据集，为了从头开始
X = np.vstack((X_train, X_test)) #按列合并
y = np.hstack((y_train, y_test)) #按行合并
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

# onr-hot 编码
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

# 设置超参数
epochs = 50
batch_size = 1

model.add(Conv2D(32, kernel_size = (3, 3), strides = 1 , padding = 'same', activation='relu', input_shape = (28, 28, 1)))
model.add(Conv2D(64, kernel_size = (3, 3), strides = 1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides =2, padding = 'valid'))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(256, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding = 'valid'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

sgd = SGD(decay=1e-6, lr=0.01, momentum=0.9, nesterov=True)# 这几个参数要好好研究

model.compile(loss='categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['mae', 'acc']
             )
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1, validation_split = 0.1)