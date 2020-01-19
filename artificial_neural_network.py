import os
import struct
import sys

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
    """ Feedforward neural network / Multi-layer perceptron classifier.
    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.
    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.
    """

    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation
        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.
        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.
        """
        onehot是10行（y有多少就多少）列，每一行分别代表这10个数字其中一个在这y中的位置
        转置后onehot.T就变成 y有多少就多少）行 10列，哪个索引上是1就代表这一行的数字是几
        热编码的意思 https://blog.csdn.net/qq_27825451/article/details/83823665
        """
        #
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)
        Returns
        ---------
        cost : float
            Regularized cost
        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term

        # If you are applying this cost function to other
        # datasets where activation
        # values maybe become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)

        return cost

    def predict(self, X):
        """Predict class labels
        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.
        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples]
            Sample labels for validation during training
        Returns:
        ----------
        self
        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################
        """    
          loc：float
              此概率分布的均值（对应着整个分布的中心centre）
          scale：float
              此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
          size：int or tuple of ints
              输出的shape，默认为None，只输出一个值
        """

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        # print(self.b_h)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))
        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))
        epoch_strlen = len(str(self.epochs))  # for progress formatting
        # print(self.epochs, epoch_strlen)
        self.eval_ = {"cost": [], "train_acc": [], "valid_acc": []}
        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):
            # iterate over minibatches
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)  # 每次都把[0,1,2...X_train.shape[0]-1, X_train.shape[0]]里的数字顺序给打乱
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                # print("start_idx: ", start_idx + self.minibatch_size)
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                # forward propagation
                #
                """
                    indices 是一批 [0,1,2...X_train.shape[0]-1, X_train.shape[0]] 的数字（做索引用）
                    batch_idx 是 indices 随机取的 minibatch_size 个数字（做索引用）
                    X_train[batch_idx] 每次只传 minibatch_size 个进去训练
                  y_train_enc[batch_idx] 
                """
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])  # 前向传播

                ##################
                # Backpropagation 反向传播训练
                ##################

                # [n_samples, n_classlabels]
                # print(y_train_enc.shape, batch_idx, y_train_enc[batch_idx])
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                # print("sigma_out.shape: ", sigma_out.shape, "self.w_out.T", self.w_out.T.shape)
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)
                # print("sigma_h: ", sigma_h.shape)
                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write("\r%0*d/%d | Cost: %.2f "
                             "| Train/Valid Acc.: %.2f%%/%.2f%% " %
                             (epoch_strlen, i + 1, self.epochs, cost,
                              train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            self.eval_["cost"].append(cost)
            self.eval_["train_acc"].append(train_acc)
            self.eval_["valid_acc"].append(valid_acc)

        return self


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
        # print(mnist.files)
        x_train, y_train, x_test, y_test = [mnist[f] for f in mnist.files]
        # print(y_train[500:800])
        nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0001, minibatch_size=100, shuffle=True, seed=1)
        nn.fit(X_train=x_train[:5000], y_train=y_train[:5000], X_valid=x_train[5000:6000], y_valid=y_train[5000:6000])
        # plt.plot(range(nn.epochs), nn.eval_["cost"])
        # plt.xlabel("Epochs")
        # plt.ylabel("Cost")
        # plt.show()
        # plt.plot(range(nn.epochs), nn.eval_["train_acc"], label="training")
        # plt.plot(range(nn.epochs), nn.eval_["valid_acc"], label="validation", linestyle="--")
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.show()
        y_test_pred = nn.predict(x_test)
        acc = (np.sum(y_test == y_test_pred).astype(np.float) / x_test.shape[0])
        print("Training accuracy: %.2f%%" % (acc * 100))
    elif execute == 3:
        pass
        a = np.array([[2, 2], [3, 3]])
        b = np.array([[2, 2], [3, 3]])
        print(a * b)
        print(np.dot(a, b))
