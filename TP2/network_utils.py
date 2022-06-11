# ##############################################################################
#  Aprendizagem Profunda, TP2 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################

from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Dropout


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
