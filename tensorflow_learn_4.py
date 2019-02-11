#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

x_data = np.linspace(-5, 5, 11)[:, np.newaxis]
print(x_data)
# noise = np.random.normal(0, 0.02, x_data.shape)
# # print(noise)
y_data = np.square(x_data)
print(y_data)
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

Weights_L1 = tf.Variable(tf.random.normal[1, 10])
biases_L1 = tf.Variable(tf.zeros[1, 10])
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)