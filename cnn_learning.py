import os
import warnings
import imageio
import numpy as np
import tensorflow as tf
from artificial_neural_network import load_mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)


def batch_generator(x, y, batch_size=64, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        x = x[idx]
        y = y[idx]
    for i in range(0, x.shape[0], batch_size):
        yield (x[i:i + batch_size, :], y[i:i + batch_size])


def conv_layer(input_tensor, name, kernel_size, n_output_channels, padding_mode="SAME", strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)  # padding 填充
        print(conv)
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

        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            conv_layer(x, name="convtest", kernel_size=(3, 3), n_output_channels=32)
        del g, x
