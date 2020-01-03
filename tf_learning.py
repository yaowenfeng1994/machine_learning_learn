import os
import warnings

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)

# hello = tf.constant('Hello, Tensorflow!')
# sess = tf.Session()
# print(sess.run(hello))
# a = tf.constant(5)
# b = tf.constant(2)
# print(sess.run(a + b))

if __name__ == "__main__":
    execute = 2
    if execute == 1:
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=None, name="x")
            w = tf.Variable(2.0, name="weight")
            b = tf.Variable(0.7, name="bias")
            z = w * x + b
            init = tf.global_variables_initializer()
        with tf.Session(graph=g) as sess:
            sess.run(init)
            for t in [1.0, 0.6, -1.8]:
                print(t, sess.run(z, feed_dict={x: t}))
    elif execute == 2:
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="input_x")
            x2 = tf.reshape(x, shape=(-1, 6), name="x2")
            xsum = tf.reduce_sum(x2, axis=0, name="col_sum")
            xmean = tf.reduce_mean(x2, axis=0, name="col_mean")
        with tf.Session(graph=g) as sess:
            x_array = np.arange(18).reshape(3, 2, 3)
            print(x_array)
            # print("input shape ", x_array.shape)
            # print("Reshape: \n", sess.run(x2, feed_dict={x: x_array}))
            # print("Column Sums: \n", sess.run(xsum, feed_dict={x: x_array}))
            # print("Column Means: \n", sess.run(xmean, feed_dict={x: x_array}))
        # print(x_array)
