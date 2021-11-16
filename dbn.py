import math
import numpy as np
import numpy as np
import pickle
from rbm import *
from utils import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid


class DBN:
    def __init__(self, n_v, layers, k=1, lr=0.01, mb_size = 1):
        """ the DBN class
        Args:
            n_v: the visible layer dimension
            layers: a list, the dimension of each hidden layer, e.g,, [500, 784]
            k: the number of gibbs sampling steps for each RBM
        """

        # self.rbm_1 = rbm.RBM(784, n_hidden=784, k=k, lr=lr, minibatch_size=mb_size)
        # self.rbm_2 = rbm.RBM(784, n_hidden=500, k=k, lr=lr, minibatch_size=mb_size)

        self.rbm_1 = rbm.RBM( n_visible = 76, n_hidden= 1056, k=k, lr=lr, minibatch_size=mb_size)

        self.rbm_2 = rbm.RBM(n_visible = 1056, n_hidden= 576, k=k, lr=lr, minibatch_size=mb_size)

        self.n_v = n_v
        self.layers = layers
        self.k = k
        self.lr = lr
        self.mb_size = mb_size

    def train(self, train_data, test_data, epochs=100):
        """ The training process of a DBN, basically we train RBMs one by one
        Args:
            train_data: the train images, numpy matrix
            valid_data: the valid images, numpy matrix
            epochs: the trainig epochs for each RBM
            lr: learning rate
        """

        # Build a DBN with two hidden layers with 500 and 784 units respectively, so there are two RBMs with 500 and 784 hidden units.

        # Errors over epochs
        train_errors_rbm1, test_errors_rbm1 = [], []
        train_errors_rbm2, test_errors_rbm2 = [], []

        # train_error_rbm1, test_error_rbm1 = self.rbm_1.reconstruction_error_epoch(
        #     train_data, test_data)
        # train_error_rbm2, test_error_rbm2 = self.rbm_2.reconstruction_error_epoch(
        #     self.rbm_1.v_sample, test_data)
        #
        # train_errors_rbm1.append(train_error_rbm1)
        # test_errors_rbm1.append(test_error_rbm1)
        #
        # train_errors_rbm2.append(train_error_rbm2)
        # test_errors_rbm2.append(test_error_rbm2)
        #
        # print('initial train error rmb 1', round(train_errors_rbm1[0], 4))
        # print('initial validation error rmb 1', round(test_errors_rbm1[0], 4))

        # print('initial train error rmb 2', round(train_errors_rbm2[0], 4))
        # print('initial validation error rmb 2', round(test_errors_rbm2[0], 4))


        for epoch in range(epochs):

            print('Epoch: ', epoch)

            t0 = time.time()

            # if epoch == 0:
            #     train_errors_rbm1, test_errors_rbm1 = self.rbm_1.train(train_data, test_data, epochs = 1)
            # else:
            #     train_errors_rbm1, test_errors_rbm1 = self.rbm_1.train((rbm_2.v_sample, test_data, epochs=1)

            train_error_rbm1, test_error_rbm1 = self.rbm_1.train(train_data, test_data, epochs=1, silent_mode = True)

            train_error_rbm2, test_error_rbm2 = self.rbm_2.train(self.rbm_1.h_sample, self.rbm_1.h_sample, epochs=1, silent_mode = True)

            train_errors_rbm1.append(train_error_rbm1[-1])
            test_errors_rbm1.append(test_error_rbm1[-1])

            train_errors_rbm2.append(train_error_rbm2[-1])
            test_errors_rbm2.append(test_error_rbm2[-1])

            print('train error rmb 1', round(train_errors_rbm1[-1], 4))
            print('validation error rmb 1', round(test_errors_rbm1[-1], 4))

            print('train error rmb 2', round(train_errors_rbm2[-1], 4))
            print('validation error rmb 2', round(test_errors_rbm2[-1], 4))

            print('time: ', time.time() - t0)

            self.rbm_1.h_sample.shape
            od_pred = self.rbm_2.h_sample

            self.rbm_2.h_sample

            self.rbm_1.W

        return train_errors_rbm1, train_errors_rbm2, test_errors_rbm1, test_errors_rbm2, od_pred

    def sample_v(self, v, k):

        v_rbm2 = self.rbm_2.gibbs_k(v = v, k = k)[3]
        v_rbm1 = self.rbm_1.sample_v(self.rbm_1.sample_h(v_rbm2)[0])[0]

        return v_rbm1

        # v_rbm1 = self.rbm_1.gibbs_k(v=v_rbm2, k=k)[3]
        # v_rbm1 = self.rbm_1.sample_v(v_rbm2[np.newaxis,:])[0]

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load Fashion MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

if __name__ == "__main__":

    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='data, parameters, etc.')
    parser.add_argument('-train', type=str, help='training file path', default='../data/digitstrain.txt')
    parser.add_argument('-valid', type=str, help='validation file path', default='../data/digitsvalid.txt')
    parser.add_argument('-test', type=str, help="test file path", default="../data/digitstest.txt")
    parser.add_argument('-max_epoch', type=int, help="maximum epoch", default=50)

    # parser.add_argument('-n_hidden', type=int, help="num of hidden units", default=250)
    parser.add_argument('-n_hidden', type=int, help="num of hidden units", default=100)
    # parser.add_argument('-k', type=int, help="CD-k sampling", default=3)
    parser.add_argument('-k', type=int, help="CD-k sampling", default=1)
    parser.add_argument('-lr', type=float, help="learning rate", default=0.01)
    parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=1)

    args = parser.parse_args()

    images, labels= load_mnist('../data/', 't10k')
    data = np.concatenate((images, labels.reshape(-1 ,1)), axis = 1)
    np.random.shuffle(data)
    train_X = data[:3000, :-1] / 255
    train_X = binary_data(train_X)
    train_Y = data[:3000, -1]

    valid_X = data[3000:4000, :-1] / 255
    valid_Y = data[3000:4000, -1]
    valid_X = binary_data(valid_X)

    # test_X = data[4000:7000, :-1] / 255
    # test_X = binary_data(test_X)
    # test_Y = data[4000:7000, -1]

    n_visible = train_X.shape[1]

    print("input dimension is " + str(n_visible))

    # Truncate our dataset and only retain images of coats (4), sandals (5), and bags (8).

    subset_idxs = []
    counter = 0
    for i in train_Y:
        if i in [4,5,8]:
            subset_idxs.append(counter)
        counter += 1

    train_X = train_X[subset_idxs,:]
    train_Y = train_Y[subset_idxs]

    subset_idxs = []
    counter = 0
    for i in valid_Y:
        if i in [4, 5, 8]:
            subset_idxs.append(counter)
        counter += 1

    valid_X = valid_X[subset_idxs, :]
    valid_Y = valid_Y[subset_idxs]


    # # Select subset of images in the meantime to debug training at scale
    # n = train_X.shape[0]
    # # n = 2000
    # # n = 100
    #
    # idxs_initial_train = np.random.choice(train_X.shape[0], n, replace=False)
    # idxs_initial_valid = np.random.choice(valid_X.shape[0], n, replace=False)
    #
    # # Select a random sample of the training data and labels
    # train_X = train_X[idxs_initial_train,:]
    # train_Y = train_Y[idxs_initial_train]
    #
    # valid_X = valid_X[idxs_initial_valid,:]
    # valid_Y = valid_Y[idxs_initial_valid]

    # idxs = np.arange(0, n)

    # General parameters
    epochs = 50
    mb_size = 64  #32 #32


    ####################################################################################
    # a) Training (5 points) Train this DBN using k = 3 for the Gibbs sampler. For each RBM, plot reconstruction error against the epoch number for training and validation on one plot. So you should include 2 plots here, each containing two curves for training and validation.
    ####################################################################################

    np.random.seed(2021)

    print('DBN Task A')

    dbn = DBN(n_v = 784, layers = [784, 100], k=3, lr=0.01, mb_size=mb_size)

    train_errors_rbm1, train_errors_rbm2, valid_errors_rbm1, valid_errors_rbm2 = dbn.train(train_X, valid_X, epochs = epochs)

    # Plot RBM1
    epochs_x = np.arange(len(train_errors_rbm1))

    fig = plt.figure()

    plt.plot(epochs_x,  train_errors_rbm1, label="Train error (RBM1)", color='black')
    plt.plot(epochs_x, valid_errors_rbm1, label="Validation error (RBM1)", color='blue')

    plt.xlabel('epoch')
    plt.ylabel('reconstruction error')
    plt.legend()

    plt.savefig('figures/dbn/errors_dbn_rmb1.pdf')

    # Plot RBM2
    epochs_x = np.arange(len(train_errors_rbm2))

    fig = plt.figure()

    plt.plot(epochs_x, train_errors_rbm2, label="Train error (RBM2)", color='black')
    plt.plot(epochs_x, valid_errors_rbm2, label="Validation error (RBM2)", color='blue')

    plt.xlabel('epoch')
    plt.ylabel('reconstruction error')
    plt.legend()

    plt.savefig('figures/dbn/errors_dbn_rbm2.pdf')


    ####################################################################################
    # b) Generation (5 points) Generate samples from the DBN by sampling from the undirected layer using k=1000, then sampling the subsequent layers as a standard directed layer. Display 100 generated samples in this way. Do they look like clothes/fashion items?
    ####################################################################################

    print('DBN Task B')

    n = 100

    idxs_initial_test = np.random.choice(valid_X.shape[0], n, replace=False)

    # Select a random sample of the test set
    test_sample = valid_X[idxs_initial_test,:]

    for index in range(test_sample.shape[0]):
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.1,
                         share_all=True
                         )

        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])

        original_image = test_sample[index,:].reshape((28,28))
        reconstructed_image = dbn.sample_v(test_sample[index][np.newaxis,:], k = 1000).reshape((28,28))

        grid[0].imshow(original_image)
        grid[1].imshow(reconstructed_image)

        plt.savefig('figures/dbn/taskB/testing_errors_task3_'+str(index) + '.pdf')

        plt.close(fig)


    reconstructed_images = np.zeros((n, 28,28))
    original_images = np.zeros((n, 28, 28))
    # index = 1
    for index in range(test_sample.shape[0]):
        original_images[index] = test_sample[index,:].reshape((28,28))
        reconstructed_images[index] = dbn.sample_v(test_sample[index][np.newaxis,:], k = 1000).reshape((28,28))

    img_tile(reconstructed_images, path =  os.getcwd() + '/figures/dbn', filename = 'dbn_reconstructed_images', save = True, aspect_ratio=1.0, border=1, border_color=0)

    img_tile(original_images, path =  os.getcwd() + '/figures/dbn', filename = 'dbn_original_images', save = True, aspect_ratio=1.0, border=1, border_color=0)

