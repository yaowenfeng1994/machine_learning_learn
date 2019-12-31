import os
import struct

import matplotlib.pyplot as plt
import numpy as np


def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
    image_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)
    with open(labels_path, "rb") as lbpath:
        struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path, "rb") as imgpath:
        struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels


class NeuralNetMLP(object):
    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size


if __name__ == "__main__":
    execute = 2
    if execute == 1:
        x_train, y_train = load_mnist("MNIST_data")
        print("Train Rows: %d, columns: %d" % (x_train.shape[0], x_train.shape[1]))
        x_test, y_test = load_mnist("MNIST_data", kind="t10k")
        print("Test Rows: %d, columns: %d" % (x_test.shape[0], x_test.shape[1]))
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(25):
            img = x_train[y_train == 7][i].reshape(28, 28)
            ax[i].imshow(img, cmap="Greys")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        np.savez_compressed("mnist_scaled.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    elif execute == 2:
        mnist = np.load("mnist_scaled.npz")
        print(mnist.files)
        x_train, y_train, x_test, y_test = [mnist[f] for f in mnist.files]
        print(x_train)

    elif execute == 3:
        pass
