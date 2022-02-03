import numpy as np
import matplotlib.pyplot as plt
# remove cv2 before you submit to autograder
import cv2
import os
from PIL import Image

class Activation:

    def __call__(self, inp):
        return self.forward(inp)

    def forward(self, inp):
        pass

    def backward(self, inp):
        pass


class Sigmoid(Activation):

    def forward(self, inp):
        self.y = 1 / (1 + np.exp(-inp))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1 - self.y)


class Relu(Activation):

    def forward(self, inp):
        self.bool = inp > 0
        return inp * self.bool

    def backward(self, grad):
        return grad * self.bool

class LRelu(Activation):
    def __init__(self, model):
        assert model == "wgan" or model == "VAE"
        self.alpha = 0.2 if model == "wgan" else 0.01

    def forward(self, inp):
        self.x = inp
        return np.maximum(inp, inp * self.alpha)

    def backward(self, grad):
        dx = np.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return dx * grad


class Tanh(Activation):

    def forward(self, inp):
        self.inp = np.tanh(inp)
        return np.tanh(inp)

    def backward(self, grad):
        return grad*(1.0 - self.inp ** 2)


def img_tile(imgs, path, filename, save, aspect_ratio=1.0, border=1, border_color=0):
    """
    Visualize the WGAN result for each step
    :param imgs: Numpy array of the generated images
    :param path: Path to save visualized results for each epoch
    :param epoch: Epoch index
    :param save: Boolean value to determine whether you want to save the result or not
    """

    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    tile_shape = None
    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break

            # -1~1 to 0~1
            img = (imgs[img_idx] + 1) / 2.0  # * 255.0

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    # path_name = path + "/epoch_%03d" % (epoch) + ".jpg"
    path_name = path + "/" + filename + ".jpg"

    ##########################################
    # Change code below if you want to save results using PIL
    ##########################################
    tile_img = cv2.resize(tile_img, (256, 256))
    cv2.imshow("Results", tile_img)
    cv2.waitKey(1)
    if save:
        cv2.imwrite(path_name, tile_img * 255)
        print('Saving image')


def mnist_reader(numbers):
    """
    Read MNIST dataset with specific numbers you needed
    :param numbers: A list of number from 0 - 9 as you needed
    :return: A tuple of a numpy array with specific numbers MNIST training dataset,
             labels of the training set and the length of the training dataset.
    """
    # Training Data
    f = open('./data/train-images.idx3-ubyte')
    loaded = np.fromfile(file=f, dtype=np.uint8)
    trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) / 127.5 - 1

    f = open('./data/train-labels.idx1-ubyte')
    loaded = np.fromfile(file=f, dtype=np.uint8)
    trainY = loaded[8:].reshape((60000)).astype(np.int32)

    _trainX = []
    for idx in range(0, len(trainX)):
        if trainY[idx] in numbers:
            _trainX.append(trainX[idx])
    return np.array(_trainX), trainY, len(_trainX)

def BCE_loss(x, y):
    """
    Binary Cross Entropy Loss for VAE
    """
    epsilon = 10e-8
    loss = np.sum(-y * np.log(x + epsilon) - (1 - y) * np.log(1 - x + epsilon))
    return loss

def img_save(imgs, path, epoch):
    """
    Save the generated images for each epoch for VAE
    :param imgs: (batch_size, 28, 28)
    :param path: path to save the imgs
    :param epoch: # of epoch
    :return:
    """
    aspect_ratio = 1.0
    border = 1
    border_color = 0
    if not os.path.exists(path):
        os.mkdir(path)
    img_num = imgs.shape[0] # 64 batch_size

    # Grid-like images
    img_shape = np.array(imgs.shape[1:3])
    img_aspect_ratio = img_shape[1] / float(img_shape[0])
    aspect_ratio *= img_aspect_ratio
    tile_height = int(np.ceil(np.sqrt(img_num * aspect_ratio)))
    tile_width = int(np.ceil(np.sqrt(img_num / aspect_ratio)))
    grid_shape = np.array((tile_height, tile_width))

    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= img_num:
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img
    file_name = path + "/iteration_{}.png".format(epoch)
    img = Image.fromarray(np.uint8(tile_img * 255), 'L')
    img.save(file_name)
