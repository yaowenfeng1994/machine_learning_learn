#! -*- coding:utf-8 -*-
# https://www.bilibili.com/video/av38019734/?p=8&t=1298

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

x_data = np.linspace(-0.5, 0.5, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 矩阵乘法前者的列数要等于后者的行数,才能相乘
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

init = tf.global_variables_initializer()

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# x1 = tf.constant([[1.], [2.]])
# y1 = tf.constant([[1., 2., 3.]])
# z1 = tf.matmul(x1, y1)
with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(z1))
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, "r-", lw=5)
    plt.show()
