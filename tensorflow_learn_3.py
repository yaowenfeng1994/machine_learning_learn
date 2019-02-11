#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 为什么随机数不能用整数？randint
x_data = np.random.rand(100)
print(111, x_data)
y_data = x_data * 10 + 20

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b
print(222, y)

s = tf.square(y_data - y)
print(333, s)
loss = tf.reduce_mean(s)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(3001):
        sess.run(train)
        if step%200 == 0:
            print(step, sess.run([k, b]))
