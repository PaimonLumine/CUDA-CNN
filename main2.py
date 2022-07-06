import tensorflow as tf
import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(tf.config.list_physical_devices('GPU'))
# In this tutorial, we will be training a lot of models. In order to use GPU memory cautiously,
# we will set tensorflow option to grow GPU memory allocation when required.
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

mymodel = load_model('mdl/model2')

# 数据集
(_, _), (X_test, y_test) = mnist.load_data()  # 划分MNIST训练集、测试集

while True:

   # 随机数
   index = random.randint(0, X_test.shape[0])
   x = X_test[index]
   y = y_test[index]

    # 显示该数字
   plt.imshow(x, cmap='gray_r')
   plt.title("original {}".format(y))
   plt.show()

   z = mymodel.predict(x)
   print('predict:', z)