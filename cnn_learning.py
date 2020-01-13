import os
import warnings

import imageio
import numpy as np
import tensorflow as tf

from artificial_neural_network import load_mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)


# https://blog.csdn.net/weixin_38368941/article/details/80000447


def batch_generator(x, y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        x = x[idx]
        y = y[idx]
    for i in range(0, x.shape[0], batch_size):
        yield (x[i:i + batch_size, :], y[i:i + batch_size])


# kernel_size 核张量维度  n_output_channels 输出特征分布图的数量
def conv_layer(input_tensor, name, kernel_size, n_output_channels, padding_mode="SAME", strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):  # 变量作用域
        input_shape = input_tensor.get_shape().as_list()
        print("input_shape: ", input_shape)
        n_input_channels = input_shape[-1]
        print("n_input_channels:", n_input_channels)
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        print("weights_shape: ", weights_shape)  # [5, 5, 1, 32] 同理卷积层也是5*5*1的图像有32个卷积核，所以一共有5*5*32个权重
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)  # padding 填充
        print(conv)  # [?, 24, 24, 32] 24*24 是原 28*28 经过卷积核缩小后的样式
        conv = tf.nn.bias_add(conv, biases, name="net_pre-activation")
        print(conv)
        conv = tf.nn.relu(conv, name="activation")
        print(conv)
        return conv


def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_units]))
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases, name="net_pre-activation")
        print(layer)
        if activation_fn is None:
            return layer
        layer = activation_fn(layer, name="activation")
        print(layer)
        return layer


def build_cnn(learning_rate=.001):
    tf_x = tf.placeholder(tf.float32, shape=[None, 784], name="tf_x")
    tf_y = tf.placeholder(tf.int32, shape=[None], name="tf_y")
    # 为什么有四个维度，因为图像还有一个深度，比如颜色有RGB，第一个-1代表未知多少张图片，28*28，最后一个维度1代表颜色（也可以理解为深度）
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name="tf_x_reshaped")
    print(tf_x_image)
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name="tf_y_onehot")
    print("\nBuilding 1st layer: ")
    h1 = conv_layer(tf_x_image, name="conv_1", kernel_size=(5, 5), padding_mode="VALID", n_output_channels=32)
    h1_pool = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print("\nBuilding 2nd layer: ")
    h2 = conv_layer(h1_pool, name="conv_2", kernel_size=(5, 5), padding_mode="VALID", n_output_channels=64)
    h2_pool = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print("\nBuilding 3rd layer: ")
    h3 = fc_layer(h2_pool, name="fc_3", n_output_units=1024, activation_fn=tf.nn.relu)
    keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name="dropout_layer")
    print("\nBuilding 4th layer: ")
    h4 = fc_layer(h3_drop, name="fc_4", n_output_units=10, activation_fn=None)
    predictions = {
        "probabilities": tf.nn.softmax(h4, name="probabilities"),
        "labels": tf.cast(tf.argmax(h4, axis=1), tf.int32, name="labels")
    }
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot),
                                        name="cross_entropy_loss")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name="train_op")
    correct_predictions = tf.equal(predictions["labels"], tf_y, name="correct_preds")
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


def save(saver, sess, epoch, path="./model/"):
    if not os.path.isdir(path):
        os.makedirs(path)
    print("Saving model in %s " % path)
    saver.save(sess, os.path.join(path, "cnn-model.ckpt"), global_step=epoch)


def load(saver, sess, path, epoch):
    print("Loading model in %s " % path)
    saver.restore(sess, os.path.join(path, "cnn-model.ckpt-%d" % epoch))


def train(sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True, dropout=0.5,
          random_seed=None):
    x_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []
    if initialize:
        sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)
    for epoch in range(1, epochs + 1):
        batch_gen = batch_generator(x_data, y_data, shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {"tf_x:0": batch_x, "tf_y:0": batch_y, "fc_keep_prob:0": dropout}
            loss, _ = sess.run(["cross_entropy_loss:0", "train_op"], feed_dict=feed)
            avg_loss += loss
            training_loss.append(avg_loss / (i + 1))
        print("Epoch %02d Training Avg. Loss: %7.3f" % (epoch, avg_loss), end=" ")
        if validation_set is not None:
            feed = {"tf_x:0": validation_set[0], "tf_y:0": validation_set[1], "fc_keep_prob:0": 1.0}
            valid_acc = sess.run("accuracy:0", feed_dict=feed)
            print("Validation Acc: %7.3f" % valid_acc)
        else:
            print()


def predict(sess, x_test, return_proba=False):
    feed = {"tf_x:0": x_test, "fc_keep_prob:0": 1.0}
    if return_proba:
        return sess.run("probabilities:0", feed_dict=feed)
    else:
        return sess.run("labels:0", feed_dict=feed)


if __name__ == "__main__":
    execute = 2
    if execute == 1:
        img = imageio.imread("dog.jpg", pilmode="RGB")
        print("Image shape:", img.shape)
        print("Number of channels:", img.shape[2])
        print("Image data type:", img.dtype)
    elif execute == 2:
        x_data, y_data = load_mnist("MNIST_data", kind="train")
        print("Rows: %d, Columns: %d" % (x_data.shape[0], x_data.shape[1]))
        x_test, y_test = load_mnist("MNIST_data", kind="t10k")
        print("Rows: %d, Columns: %d" % (x_test.shape[0], x_test.shape[1]))

        x_train, y_train = x_data[:50000, :], y_data[:50000]
        x_valid, y_valid = x_data[50000:, :], y_data[50000:]
        print("Training:    ", x_train.shape, y_train.shape)
        print("Validation:  ", x_valid.shape, y_valid.shape)
        print("Test Set:    ", x_test.shape, y_test.shape)

        mean_vals = np.mean(x_train, axis=0)
        std_vals = np.std(x_train)

        x_train_centered = (x_train - mean_vals) / std_vals
        x_valid_centered = (x_valid - mean_vals) / std_vals
        x_test_centered = (x_test - mean_vals) / std_vals
        # print("==========================================")
        # g = tf.Graph()
        # with g.as_default():
        #     x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        #     conv_layer(x, name="convtest", kernel_size=(3, 3), n_output_channels=32)
        # del g, x
        # print("==========================================")
        # g = tf.Graph()
        # with g.as_default():
        #     x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        #     fc_layer(x, name="fctest", n_output_units=32, activation_fn=tf.nn.relu)
        # del g, x
        # print("==========================================")
        g = tf.Graph()
        random_seed = 123
        with g.as_default():
            tf.set_random_seed(random_seed)
            build_cnn()
            saver = tf.train.Saver()
        # with tf.Session(graph=g) as sess:
        #     train(sess, training_set=(x_train_centered, y_train), validation_set=(x_valid_centered, y_valid),
        #           initialize=True, random_seed=random_seed)
        #     save(saver, sess, epoch=20)
        with tf.Session(graph=g) as sess:
            load(saver, sess, epoch=20, path="./model")
            preds = predict(sess, x_test_centered, return_proba=False)
            print("Test Accuracy: %.3f%%" % (100 * np.sum(preds == y_test) / len(y_test)))
    elif execute == 3:
        pass
