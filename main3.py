import tensorflow as tf
import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(tf.config.list_physical_devices('GPU'))
# In this tutorial, we will be training a lot of models. In order to use GPU memory cautiously,
# we will set tensorflow option to grow GPU memory allocation when required.
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
img_rows, img_cols = 28, 28 

if K.image_data_format() == 'channels_first': 
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
   input_shape = (1, img_rows, img_cols) 
else: 
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
   input_shape = (img_rows, img_cols, 1) 
   
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

y_train = keras.utils.to_categorical(y_train, 10) 
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential() 
model.add(Conv2D(6, kernel_size = (5, 5),  
   activation = 'relu', input_shape = input_shape)) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Conv2D(16, (5, 5), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(120, activation = 'relu')) 
model.add(Dense(100, activation = 'relu')) 
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, 
   optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

model.summary()

print(x_train.shape, y_train.shape)
model.fit(
   x_train, y_train, 
   batch_size = 32,
   epochs = 1000,
   verbose = 1, 
   validation_data = (x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

model.save('mdl/model3')