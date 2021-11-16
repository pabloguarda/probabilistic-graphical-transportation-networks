import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import sys
import random
import time
import math
import argparse
import pickle
import os
from utils import *


# My imports
import rbm
from numpy.testing import assert_allclose
import pickle as pk

if not os.path.exists('../plot'):
    os.makedirs('../plot')
if not os.path.exists('../dump'):
    os.makedirs('../dump')

# seed = 10417617
np.random.seed(10417617)

def binary_data(inp):
    return (inp > 0.5) * 1.

def sigmoid(x):
    """
    Args:
        x: input

    Returns: the sigmoid of x

    """

    # x = np.clip(x, -100,100)

    return 1 / (1 + np.exp(-x))

def shuffle_corpus(data):
    """shuffle the corpus randomly
    Args:
        data: the image vectors, [num_images, image_dim]
    Returns: The same images with different order
    """
    random_idx = np.random.permutation(len(data))
    return data[random_idx]


class RBM:
    def __init__(self, n_visible, n_hidden, k, lr=0.01, minibatch_size=1):
        """The RBM base class
        Args:
            n_visible: the dimension of visible layer
            n_hidden: the dimension of hidden layer
            k: number of gibbs sampling steps
            lr: learning rate
            minibatch_size: the size of each training batch
            hbias: the bias for the hidden layer
            vbias: the bias for the visible layer
            W: the weights between visible and hidden layer
        """
        np.random.seed(10417617)
        # k is the gibbs sampling step

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = lr
        self.minibatch_size = minibatch_size

        # Initialize weight matrix
        self.W = np.random.normal(0,np.sqrt(6.0/(n_hidden+n_visible)),(n_hidden,n_visible))

        # Initialize bias
        self.hbias = np.zeros((self.n_hidden))
        self.vbias = np.zeros((self.n_visible))


        # Train or Test mode
        # self.train_mode = True


    # p(h=1|v)
    def h_v(self, v):
        """ Transform the visible vector to hidden vector and compute its probability being 1
        Args:
            v: the visible vector
                shape: (1,dim_v)
        Returns:
            The probability of output hidden vector, h, being 1, i.e., p(h=1|v)
                shape: (1,dim_h)
        """

        N = v.shape[0]

        # p_hv = np.zeros((self.n_hidden,n))
        # p_hv = np.zeros((N,self.n_hidden))

        # for n in range(N):
            # for j in range(self.n_hidden):
            #     p_hv[j][n] = sigmoid(self.hbias[j]+self.W[j,:].dot(v[n].flatten()))
            #     # p_h = p_h*p_hj

            # p_hv[n] = sigmoid(self.hbias + self.W.dot(v[n].T))

        # Vectorized operation
        p_hv = sigmoid(self.hbias + self.W.dot( v.T).T)



        return p_hv


        # return 1/(1+np.exp(-(self.vbias.T+self.W*v)))

    # sample h given h_v
    def sample_h(self, v):
        """ sample a hidden vector given the distribution p(h=1|v)
        Args:
            v: the visible vector v
                shape: (1, dim_v)
        Return:
            The sampling hidden vectors, which are binary in our experiment,
            as well as the distribution h_v
            shape: (1, dim_h)
        """

        np.random.seed(10417617)

        phv = self.h_v(v)

        n = phv.shape[0]

        # hv = np.zeros(phv.shape)
        # for i in range(phv.shape[0]):
        #     hv[i] = (phv[i]>np.random.uniform(0, 1)).astype(float)

        # for k in range(self.k):
        # hv  = (phv.flatten()>np.random.uniform(0,1,hv.size))

        hv = np.random.binomial(1, phv, size=phv.shape)

        # hv = hv.astype(int)

        return hv, phv

    # p(v=1|h)
    def v_h(self, h):
        """ Transform the hidden vector to visible vector and compute its probability being 1
        Args:
            h: the hidden vector
                shape: (1,dim_h)
        Returns:
            The probability of output visible vector, v, being 1, i.e., p(v=1|h)
                shape: (1,dim_v)
        """
        np.random.seed(10417617)

        # n = h.shape[0]

        # p_vh = np.zeros((n,self.n_visible))

        # for n in range(n):
        #     for i in range(self.n_visible):
        #         p_vh[n,i] = sigmoid(self.vbias[i]+self.W[:,i].dot(h[n].flatten()))
                # p_h = p_h*p_hj

        # Vectorized operation
        p_vh = sigmoid(self.vbias + self.W.T.dot(h.T).T)

        return p_vh

    # sample v given v_h
    def sample_v(self, h):
        """ sample a visible vector given the distribution p(v=1|h)
        Args:
            h: the hidden vector h
                shape: (1,dim_h)
        Return:
            The sampling visible vectors, which are binary in our experiment
                shape: (1,dim_v)
        """

        pvh = self.v_h(h)

        vh = np.random.binomial(1, pvh, size=pvh.shape)

        return vh, pvh

    # gibbs sampling with k step
    def gibbs_k(self, v, k=None):
        """ The contrastive divergence k (CD-k) procedure
        Args:
            v: the input visible vector
                shape: (1,dim_v)
            k: the number of gibbs sampling steps
                shape: scalar (int)
        Return:
            h0: the hidden vector sample with one iteration
                shape: (1,dim_h)
            v0: the input v
                shape: (1,dim_v)
            h_sample: the hidden vector sample with k iterations
                shape: (1,dim_h)
            v_sample: the visible vector samplg with k iterations
                shape: (1,dim_v)
            prob_h: the prob of hidden being 1 after k iterations
                shape: (1,dim_h)
            prob_v: the prob of visible being 1 after k itersions
                shape: (1,dim_v)
        """
        np.random.seed(10417617)

        # Based on piazza post @156
        if k is None:
            k = self.k

        v0 = v #v[np.newaxis,:]
        h0 = self.sample_h(v0)[0]
        # v0 = self.sample_v(h0)[0]

        v_sample = v0
        h_sample = h0

        for iter in range(k):
            v_sample, prob_v = self.sample_v(h_sample)
            h_sample, prob_h = self.sample_h(v_sample)






        return h0, v0, h_sample, v_sample, prob_h, prob_v




    def update(self, x, train_mode = True):
        """ update our RBM with input x
        Args:
            x: the input data x
                shape: (1, dim_v)
        """
        np.random.seed(10417617)

        if len(x.shape) == 1:
            x = x[np.newaxis,:]

        # if train_mode == False:
        #     k = 0

        if train_mode == False:
            k = 1

        else:
            k = self.k

        # Gibbs sampling
        h0, v0, self.h_sample, self.v_sample, prob_h, prob_v = self.gibbs_k(x, k = k)

        prob_h0 = self.h_v(v0)

        # h0.shape
        # prob_h.shape
        # v0.shape

        if train_mode:
            # Update parameters
            self.W = self.W  + self.lr * (prob_h0.T.dot(v0)-prob_h.T.dot(self.v_sample))
            # self.hbias = (self.hbias[:,np.newaxis] + self.lr * (prob_h0.T - prob_h.T)).flatten()
            # self.vbias = (self.vbias[:,np.newaxis] + self.lr * (v0.T - self.v_sample.T)).flatten()
            self.hbias = np.mean((self.hbias[:,np.newaxis] + self.lr * (prob_h0.T - prob_h.T)),axis = 1)
            self.vbias = np.mean((self.vbias[:,np.newaxis] + self.lr * (v0.T - self.v_sample.T)), axis = 1)
            # pass



    # calculate ce and re loss
    def eval(self, X):
        """ Computer reconstruction error
        Args:
            X: the input X
                shape: [num_X, dim_X]
        Return:
            The reconstruction error
                shape: a scalar
        """

        # self.train_mode = False

        # np.random.seed(10417617)

        n = X.shape[0]
        # cols = X.shape[1]

        error = 0

        self.update(X, train_mode=True)

        error = X-self.v_sample
        error = np.mean(np.sqrt(np.sum(error**2,axis = 1)))


        return error

    def reconstruction_error_epoch(self, trainX, testX):

        # Compute training and validation error

        error_epoch_train = self.eval(trainX)
        # accuracy_train = loss_function.getAccu()

        error_epoch_test = self.eval(testX)
        # accuracy_test = loss_function.getAccu()

        # Normalize losses by nunmber of samples
        # error_epoch_train =  error_epoch_train/trainX.shape[0]
        # loss_epoch_test = loss_epoch_test /testX.shape[0]

        return error_epoch_train, error_epoch_test  # ,accuracy_train,accuracy_test

    def train(self, trainX, testX, epochs = 1, silent_mode = False):
        # Losses and accuracies
        train_errors, test_errors = [], []
        # train_accuracies,test_accuracies = [],[]

        # Initial loss and accuracy
        train_error, test_error = self.reconstruction_error_epoch(
            trainX, testX)

        train_errors.append(train_error)
        test_errors.append(test_error)
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)

        # print('initial train accuracy', round(train_accuracies[0],4))
        # print('initial test accuracy', round(test_accuracies[0],4))

        if not silent_mode:
            print('initial train error', round(train_errors[0], 4))
            print('initial validation error', round(test_errors[0], 4))

        for epoch in np.arange(1, epochs + 1):

            if not silent_mode:
                print('Epoch: ', epoch)

            t0 = time.time()

            if self.minibatch_size is None:
                self.minibatch_size = trainX.shape[0]

            n = trainX.shape[0]

            # Shuffle dataset at the begining of the epoch
            idxs = np.arange(0, n)

            np.random.shuffle(idxs)

            for idx in np.arange(0, n, self.minibatch_size):
                trainX_mb = trainX[idxs, :][idx:idx + self.minibatch_size, :]

                self.update(trainX_mb, train_mode = True)

            train_error, test_error = self.reconstruction_error_epoch(
                trainX, testX)

            train_errors.append(train_error)
            test_errors.append(test_error)
            # train_accuracies.append(train_accuracy)
            # test_accuracies.append(test_accuracy)

            # print('train accuracy',round(train_accuracies[epoch],4))
            # print('test accuracy', round(test_accuracies[epoch],4))

            if not silent_mode:
                print('train error', round(train_errors[epoch], 4))
                print('validation error', round(test_errors[epoch], 4))

                print('time: ', time.time() - t0)

            # print(loss_epoch)

        return train_errors, test_errors  # ,train_accuracies, test_accuracies


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

    # # Tests
    #
    seed = 10417617
    TOLERANCE = 1e-5

    with open('tests.pk', "rb") as f:
        tests = pk.load(f);

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



    # # Select subset of images in the meantime to debug training at scale
    # n = train_X.shape[0]
    # # n = 2000
    # # n = 100
    #
    # n = 200
    #
    # idxs_initial_train = np.random.choice(train_X.shape[0], n, replace=False)
    # idxs_initial_test = np.random.choice(valid_X.shape[0], n, replace=False)
    #
    # # Select a random sample of the training data and labels
    # train_X = train_X[idxs_initial_train,:]
    # train_Y = train_Y[idxs_initial_train]
    #
    # valid_X = valid_X[idxs_initial_test,:]
    # valid_Y = valid_Y[idxs_initial_test]


    # idxs = np.arange(0, n)

    # General parameters
    epochs = 50
    mb_size = 64#64 #32 #32

    sl = rbm.RBM(784, n_hidden=250, k=3, lr=0.01, minibatch_size=1)
    #input
    np.random.seed(seed)
    # visible_vectors = np.random.uniform(0,1,(1,784))
    visible_vectors = np.random.randint(2, size=(784))

    sl.update(visible_vectors)
    assert_allclose(sl.W, tests["update"]['W'], atol=TOLERANCE)
    assert_allclose(sl.hbias, tests["update"]['hbias'], atol=TOLERANCE)
    assert_allclose(sl.vbias, tests["update"]['vbias'], atol=TOLERANCE)


    ################################################################
    ##################  1.1 Task 1 (10 points) #################
    ################################################################

    # Train an RBM model with 100 hidden units, starting with CD with k = 1 step

    np.random.seed(2021)

    # Choose a reasonable learning rate (e.g. 0.1 or 0.01).

    print('RBM Task 1')

    rbm_1 = rbm.RBM(784, n_hidden=100, k=1, lr=0.01, minibatch_size=mb_size)

    # For initialization use samples from a normal distribution with mean 0 and standard deviation 0.1.
    rbm_1.W = np.random.normal(0, 0.1, (rbm_1.n_hidden, rbm_1.n_visible))

    train_errors_task1,test_errors_task1 = rbm_1.train(trainX = train_X, testX = valid_X, epochs = epochs)

    # Plots
    epochs_x = np.arange(len(train_errors_task1))

    fig = plt.figure()

    plt.plot(epochs_x, train_errors_task1, label="Train error (task 1)", color='black')
    plt.plot(epochs_x, test_errors_task1, label="Validation error (task 1)", color='blue')

    plt.xlabel('epoch')
    plt.ylabel('reconstruction error')
    plt.legend()

    plt.savefig('figures/rbm/errors_task1.pdf')

    plt.close(fig)

    # Weights
    dim = int(np.ceil(np.sqrt(rbm_1.W.shape[0])))

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(dim , dim ),
                     axes_pad=0.1,
                     share_all=True
                     )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for ax, index in zip(grid, range(rbm_1.W.shape[0])):
        ax.imshow(rbm_1.W[index,:].reshape((28,28)))

    # plt.show()\

    plt.savefig('figures/rbm/weights_task1.pdf')

    plt.close(fig)

    n = rbm_1.W.shape[0]

    weights_values = np.zeros((n, 28,28))

    for index in range(weights_values.shape[0]):
        weights_values[index] = rbm_1.W[index,:].reshape((28,28))

    img_tile(weights_values, path =  os.getcwd() + '/figures/rbm',  filename = 'rbm_weights', save = True, aspect_ratio=1.0, border=1, border_color=0)

    ##################################################################################
    # 1.2 Task 2 (5 points) Try learning an RBM model with k=1, k=3 and k=5 steps in CD-k. Describe the effect of this modification on the convergence properties, as well as the generalization of the learned model. #####################################################

    print('RBMs Task 2')

    rbm_2a = rbm.RBM(784, n_hidden=100, k=1, lr=0.01, minibatch_size=mb_size)
    rbm_2a.W = np.random.normal(0, 0.1, (rbm_2a.n_hidden, rbm_2a.n_visible))
    train_errors_task2a, valid_errors_task2a = rbm_2a.train(trainX=train_X, testX=valid_X, epochs=epochs)

    rbm_2b = rbm.RBM(784, n_hidden=100, k=3, lr=0.01, minibatch_size=mb_size)
    rbm_2b.W = np.random.normal(0, 0.1, (rbm_2b.n_hidden, rbm_2b.n_visible))
    train_errors_task2b, valid_errors_task2b = rbm_2b.train(trainX=train_X, testX=valid_X, epochs=epochs)

    # Replace back by k = 5
    rbm_2c = rbm.RBM(784, n_hidden=100, k=5, lr=0.01, minibatch_size=mb_size)
    rbm_2c.W = np.random.normal(0, 0.1, (rbm_2c.n_hidden, rbm_2c.n_visible))
    train_errors_task2c, valid_errors_task2c = rbm_2c.train(trainX=train_X, testX=valid_X, epochs=epochs)

    # Plots
    epochs_x = np.arange(len(train_errors_task2a))

    fig = plt.figure()

    plt.plot(epochs_x, train_errors_task2a, label="Train error (task 2, k =1)", color='black')
    plt.plot(epochs_x, train_errors_task2b, label="Train error (task 2, k =3)", color='blue')
    plt.plot(epochs_x, train_errors_task2c, label="Train error (task 2, k =5)", color='red')

    plt.xlabel('epoch')
    plt.ylabel('reconstruction error')
    plt.legend()

    plt.savefig('figures/rbm/training_error_task2a.pdf')

    plt.close(fig)

    fig = plt.figure()

    plt.plot(epochs_x, valid_errors_task2a, label="Validation error (task 2, k =1)", color='black')
    plt.plot(epochs_x, valid_errors_task2b, label="Validation error (task 2, k =3)", color='blue')
    plt.plot(epochs_x, valid_errors_task2c, label="Validation  error (task 2, k =5)", color='red')

    plt.xlabel('epoch')
    plt.ylabel('reconstruction error')
    plt.legend()

    plt.savefig('figures/rbm/validation_error_task2a.pdf')

    plt.close(fig)

    ##################################################################################
    # 1.3 Task 3 (5 points) To qualitatively test the model performance, initialize 100 Gibbs chains with some image vectors in test set, and reconstruct them by your RBM—namely, run the Gibbs sampler for just 1 step. Display the 100 sampled images and their original. Do the reconstructed images look close to the original image? 
    ##################################################################################

    # TODO: replace back by rbm_1, just testing the impact of a higher k
    rbm_3 = rbm_1 #rbm_2a

    n = 100

    idxs_initial_test = np.random.choice(valid_X.shape[0], n, replace=False)

    # Select a random sample of the test set
    test_sample = valid_X[idxs_initial_test,:]
    # test_sample.shape
    # index = 0

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
        reconstructed_image = rbm_3.gibbs_k(test_sample[index,:])[3].reshape((28,28))

        grid[0].imshow(original_image)
        grid[1].imshow(reconstructed_image)

        plt.savefig('figures/rbm/task3/reconstruction_error_task3_'+str(index) + '.pdf')

        plt.close(fig)

    reconstructed_images = np.zeros((n, 28,28))
    original_images = np.zeros((n, 28, 28))
    # index = 1
    for index in range(test_sample.shape[0]):

        original_images[index] = test_sample[index,:].reshape((28,28))
        reconstructed_images[index] = rbm_3.gibbs_k(test_sample[index,:])[3].reshape((28,28))

    img_tile(reconstructed_images, path =  os.getcwd() + '/figures/rbm', filename = 'rbm_reconstructed_images', save = True, aspect_ratio=1.0, border=1, border_color=0)

    img_tile(original_images, path =  os.getcwd() + '/figures/rbm', filename = 'rbm_original_images', save = True, aspect_ratio=1.0, border=1, border_color=0)
