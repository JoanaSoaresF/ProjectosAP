# ##############################################################################
#  Aprendizagem Profunda, TP2 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
import os
import shutil

import imageio
import numpy as np
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from matplotlib import pyplot as plt


def create_gif(images_frames, gif_name, time=0.2):
    filenames = os.listdir(images_frames)
    filenames.sort()
    print(filenames)
    with imageio.get_writer(gif_name, mode='I', duration=time) as writer:
        for filename in filenames:
            image = imageio.imread(f"{gif_name}/{gif_name}")
            writer.append_data(image)


def convolution_stack_layer(inputs, filters, kernel_size):
    """
    Creates a convolutional stack with two convolution layers, padding "same", ReLU activation and batch normalization
    Max pooling of size 2 × 2 and same stride
    Args:
        inputs: the inputs of the layer
        filters: number of filters
        kernel_size: kernel size

    Returns: the last layer created

    """
    layer = Conv2D(filters, kernel_size, padding="same")(inputs)
    layer = Activation("relu")(layer)
    # Use batch normalization after activation so that input of following layer is standardized
    # layer = BatchNormalization()(layer)

    layer = Conv2D(filters, kernel_size, padding="same")(layer)
    layer = Activation("relu")(layer)
    # layer = BatchNormalization()(layer)
    pool = MaxPooling2D()(layer)

    return pool


def dense_block(inputs, n, dropout=True):
    """
    Creates a dense block with Relu as activation function and batch normalization
    Args:
        inputs: the previous layer in the network
        n: the number of neurons of the dense layer
        dropout: boolean to specify is a layer of dropout is to be added

    Returns:

    """
    init = HeUniform()
    layer = Dense(n, kernel_initializer=init)(inputs)
    layer = Activation("relu")(layer)
    # Use batch normalization after activation so that input of following layer is standardized
    # layer = BatchNormalization()(layer)
    if dropout:
        layer = Dropout(0.5)(layer)
        # Use dropout for dense layers
    return layer


def plot_statistics(data, title, path):
    # data to be plotted
    x = np.arange(1, len(data)+1)

    # plotting
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(title)
    plt.plot(x, data, color="green")
    plt.savefig(path)
    plt.close()


def create_folders(training_path):
    if os.path.exists(training_path):
        # Recursively delete all subfolders and sub files under the folder
        shutil.rmtree(training_path)
    os.makedirs(training_path)
