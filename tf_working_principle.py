import os
import warnings

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    execute = 2
    if execute == 1:
        g = tf.Graph()
        with g.as_default():
            t1 = tf.constant(np.pi)
            t2 = tf.constant([1, 2, 3, 4])
            t3 = tf.constant([[1, 2], [3, 4]])

            r1 = tf.rank(t1)
            r2 = tf.rank(t2)
            r3 = tf.rank(t3)

            s1 = t1.get_shape()
            s2 = t2.get_shape()
            s3 = t3.get_shape()
            print("Shapes: ", s1, s2, s3)
        with tf.Session(graph=g) as sess:
            print("Ranks: ", r1.eval(), r2.eval(), r3.eval())
    elif execute == 2:
        g = tf.Graph()
        # with g.as_default():
        #     arr = np.array([
        #         [1., 2., 3., 3.5],
        #         [4., 5., 6., 6.5],
        #         [7., 8., 9., 9.5]
        #     ])
        #     T1 = tf.constant(arr, name="T1")
        #     print(T1)
        #     s = T1.get_shape()
        #     print("shape of T1 is ", s)
        #     T2 = tf.Variable(tf.random_normal(shape=s))
        #     print(T2)
        #     T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0],)))
        #     print(T3)
        #     T4 = tf.reshape(T1, shape=[1, 1, -1], name="T4")
        #     print(T4)
        #     T5 = tf.reshape(T1, shape=[1, 3, -1], name="T5")
        #     print(T5)
        #     t5_splt = tf.split(T5, num_or_size_splits=2, axis=2, name="T8")
        #     print(t5_splt)
        # with tf.Session(graph=g) as sess:
        #     print(sess.run(T4))
        #     print(sess.run(T5))
        # with g.as_default():
        #     t1 = tf.ones(shape=(5, 1), dtype=tf.float32, name="t1")
        #     t2 = tf.zeros(shape=(5, 1), dtype=tf.float32, name="t2")
        #     print(t1, t2)
        #     t3 = tf.concat([t1, t2], axis=0, name="t3")
        #     t4 = tf.concat([t1, t2], axis=1, name="t4")
        #     print(t3, t4)
        # with tf.Session(graph=g) as sess:
        #     print(sess.run(t3))
        #     print(sess.run(t4))
        x, y = 2., 5.
        with g.as_default():
            tf_x = tf.placeholder(dtype=tf.float32, shape=None, name="tf_x")
            tf_y = tf.placeholder(dtype=tf.float32, shape=None, name="tf_y")
            print(tf_x, tf_y)
            # if x < y:
            #     res = tf.add(tf_x, tf_y, name="result_add")
            # else:
            #     res = tf.subtract(tf_x, tf_y, name="result_sub")
            res = tf.cond(tf_x < tf_y, lambda: tf.add(tf_x, tf_y, name="result_add"),
                          lambda: tf.subtract(tf_x, tf_y, name="result_sub"))
            print("Object: ", res)
        with tf.Session(graph=g) as sess:
            print("x < y: %s -> Result: " % (x < y), res.eval(feed_dict={"tf_x:0": x, "tf_y:0": y}))
            x, y = 2.0, 1.0
            print("x < y: %s -> Result: " % (x < y), res.eval(feed_dict={"tf_x:0": x, "tf_y:0": y}))
