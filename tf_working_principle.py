import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    execute = 1
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
