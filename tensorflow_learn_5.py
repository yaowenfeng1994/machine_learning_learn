#! -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import struct

# one_hot 只有0跟1
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
print(n_batch)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        for batch in range(n_batch):
            images, labels = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: images, y: labels})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter: " + str(epoch) + ", Testing Accuracy: " + str(acc))

# filename = 'train-labels-idx1-ubyte'
# binfile = open(filename, 'rb')
# buf = binfile.read()
# print(buf)
# index = 0
# struct.unpack_from('>IIII', buf, index)
# index += struct.calcsize('>IIII')
# im = struct.unpack_from('>784B', buf, index)
# index += struct.calcsize('>784B')
# im = np.array(im)
# im = im.reshape(28, 28)
# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(im, cmap='gray')
# plt.show()
