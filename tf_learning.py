import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras

from artificial_neural_network import load_mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)


# hello = tf.constant('Hello, Tensorflow!')
# sess = tf.Session()
# print(sess.run(hello))
# a = tf.constant(5)
# b = tf.constant(2)
# print(sess.run(a + b))

class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim), name="x_input")
        self.y = tf.placeholder(dtype=tf.float32, shape=(None), name="y_input")
        print(self.X, self.y)
        w = tf.Variable(tf.zeros(shape=(1)), name="weight")
        b = tf.Variable(tf.zeros(shape=(1)), name="bisa")
        print(w, b)
        self.z_net = tf.squeeze(w * self.X + b, name="z_net")
        print(self.z_net)
        sqr_errors = tf.square(self.y - self.z_net, name="sqr_errors")
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name="mean_cost")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name="GradientDescent")
        self.optimizer = optimizer.minimize(self.mean_cost)


def train_linreg(sess, model, x_train, y_train, num_epochs=10):
    sess.run(model.init_op)
    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost], feed_dict={model.X: x_train, model.y: y_train})
        training_costs.append(cost)
    return training_costs


def predict_linreg(sess, model, x_test):
    y_pred = sess.run(model.z_net, feed_dict={model.X: x_test})
    return y_pred


def create_batch_generator(x, y, batch_size=128, shuffle=False):
    x_copy = np.array(x)
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((x_copy, y_copy))
        np.random.shuffle(data)
        x_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)
    for i in range(0, x.shape[0], batch_size):
        yield (x_copy[i: i + batch_size, :], y_copy[i: i + batch_size])


if __name__ == "__main__":
    execute = 5
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
            x_array = np.arange(18).reshape(3, 2, 3)  # 三个两行三列的矩阵
            print(x_array)
            print("input shape ", x_array.shape)
            print("Reshape: \n", sess.run(x2, feed_dict={x: x_array}))
            print("Column Sums: \n", sess.run(xsum, feed_dict={x: x_array}))
            print("Column Means: \n", sess.run(xmean, feed_dict={x: x_array}))
        # print(x_array)
    elif execute == 3:
        x_train = np.arange(10).reshape(10, 1)
        y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
        # print(x_train, y_train)
        lrmodel = TfLinreg(x_dim=x_train.shape[1], learning_rate=0.01)
        sess = tf.Session(graph=lrmodel.g)
        training_costs = train_linreg(sess, lrmodel, x_train, y_train)
        # plt.plot(range(1, len(training_costs) + 1), training_costs)
        # plt.tight_layout()
        # plt.xlabel("Epoch")
        # plt.ylabel("Training Cost")
        # plt.show()

        plt.scatter(x_train, y_train, marker="s", s=50, label="Training Data")
        plt.plot(range(x_train.shape[0]), predict_linreg(sess, lrmodel, x_train), color="gray", marker="o",
                 markersize=6, linewidth=3, label="LinReg Model")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif execute == 4:
        x_train, y_train = load_mnist("MNIST_data", kind="train")
        print("Rows: %d, Columns: %d" % (x_train.shape[0], x_train.shape[1]))
        x_test, y_test = load_mnist("MNIST_data", kind="t10k")
        print("Rows: %d, Columns: %d" % (x_train.shape[0], x_train.shape[1]))
        # axis是几，那就表明哪一维度被压缩成1
        mean_vals = np.mean(x_train, axis=0)
        std_val = np.std(x_train)
        x_train_centered = (x_train - mean_vals) / std_val
        x_test_centered = (x_test - mean_vals) / std_val
        del x_train, x_test
        print(x_train_centered.shape, y_train.shape)
        print(x_test_centered.shape, y_test.shape)

        n_features = x_train_centered.shape[1]
        n_classes = 10
        random_seed = 123
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)
            tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name="tf_x")
            tf_y = tf.placeholder(dtype=tf.int32, shape=None, name="tf_y")
            y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
            h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name="layer1")
            h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name="layer2")
            logits = tf.layers.dense(inputs=h2, units=10, activation=None, name="layer3")
            predictions = {
                "classes": tf.argmax(logits, axis=1, name="predicted_classes"),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        with g.as_default():
            cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=cost)
            init_op = tf.global_variables_initializer()
        sess = tf.Session(graph=g)
        sess.run(init_op)
        for epoch in range(50):
            training_costs = []
            # print("y_train: ", y_train, y_train.shape)
            batch_generator = create_batch_generator(x_train_centered, y_train, batch_size=64)
            for batch_x, batch_y in batch_generator:
                feed = {tf_x: batch_x, tf_y: batch_y}
                _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
                training_costs.append(batch_cost)
            print(" -- Epoch %2d Avg.Training Loss: %.4f" % (epoch + 1, np.mean(training_costs)))
        feed = {tf_x: x_test_centered}
        y_pred = sess.run(predictions["classes"], feed_dict=feed)
        print("Test Accuracy: %.2f%%" % (100 * np.sum(y_pred == y_test) / y_test.shape[0]))
    elif execute == 5:
        x_train, y_train = load_mnist("MNIST_data", kind="train")
        print("Rows: %d, Columns: %d" % (x_train.shape[0], x_train.shape[1]))
        x_test, y_test = load_mnist("MNIST_data", kind="t10k")
        print("Rows: %d, Columns: %d" % (x_train.shape[0], x_train.shape[1]))
        # axis是几，那就表明哪一维度被压缩成1
        mean_vals = np.mean(x_train, axis=0)
        std_val = np.std(x_train)
        x_train_centered = (x_train - mean_vals) / std_val
        x_test_centered = (x_test - mean_vals) / std_val
        del x_train, x_test
        print(x_train_centered.shape, y_train.shape)
        print(x_test_centered.shape, y_test.shape)

        y_train_onehot = keras.utils.to_categorical(y_train)
        print("First 3 labels: ", y_train[:3])
        print("First 3 labels (one-hot):\n", y_train_onehot[:3])
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                units=50,
                input_dim=x_train_centered.shape[1],
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                activation="tanh"
            )
        )
        model.add(
            keras.layers.Dense(
                units=50,
                input_dim=50,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                activation="tanh"
            )
        )
        model.add(
            keras.layers.Dense(
                units=y_train_onehot.shape[1],
                input_dim=50,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                activation="softmax"
            )
        )
        sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)
        model.compile(optimizer=sgd_optimizer, loss="categorical_crossentropy")
        history = model.fit(x_train_centered, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=0.1)
        y_train_pred = model.predict_classes(x_train_centered, verbose=0)
        print("First 3 predictions: ", y_train_pred[:3])
        correct_preds = np.sum(y_train == y_train_pred, axis=0)
        train_acc = correct_preds / y_train.shape[0]
        print("First 3 predictions: ", y_train_pred[:3])
        print("Training Accuracy: %.2f%%" % (train_acc * 100))
        y_test_pred = model.predict_classes(x_test_centered, verbose=0)
        correct_preds = np.sum(y_test == y_test_pred, axis=0)
        test_acc = correct_preds / y_test.shape[0]
        print("Test Accuracy: %.2f%%" % (test_acc * 100))
