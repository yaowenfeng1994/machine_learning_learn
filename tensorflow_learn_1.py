#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 用Tensorflow计算a=(b+c)∗(c+2)
# 首先，创建一个TensorFlow常量=>2
const = tf.constant(2.0, dtype=tf.float32, name="const")
# 创建TensorFlow变量b和c
# b = tf.Variable(2.0, dtype=tf.float32, name="b")
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, dtype=tf.float32, name="c")
# 创建operation
d = tf.add(b, c, name="d")
e = tf.add(c, const, name="e")
a = tf.multiply(d, e, name="a")
# 1. 定义init operation
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # 2. 运行init operation
    sess.run(init_op)
    # 计算
    # a_out = sess.run(a)
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    writer.close()
    print("Variable a is {}".format(a_out))

print(np.arange(0, 10)[:, np.newaxis])
