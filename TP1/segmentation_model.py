# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Activation, UpSampling2D, Concatenate, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

root_logdir = "logs"


def convolution_stack_layer(inputs, filters, kernel_size, polling=True):
    """
    Creates a convolutional stack with two convolution layers, padding "same", ReLU activation and batch normalization
    Max pooling of size 2 × 2 and same stride
    Args:
        polling: boolean to indicate if it uses polling in the end of the block
        inputs: the inputs of the layer
        filters: number of filters
        kernel_size: kernel size

    Returns: the last layer created

    """
    layer = Conv2D(filters, kernel_size, padding="same")(inputs)
    layer = Activation("relu")(layer)
    # Use batch normalization after activation so that input of following layer is standardized
    layer = BatchNormalization()(layer)

    layer = Conv2D(filters, kernel_size, padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)

    if polling:
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
        return layer, pool

    return layer


def up_block(inputs, conv_level, filters):
    up = UpSampling2D(size=(2, 2), interpolation="nearest")(inputs)
    up = Concatenate()([up, conv_level])
    up = Conv2D(filters, (2, 2), padding="same")(up)
    up = Activation("relu")(up)
    return up


def create_segmentation_model(input_shape):
    """
    Creates the model defined for the segmentation problem
    Args:
        input_shape: shape of the original input

    Returns: the model created

    """
    inputs = Input(shape=input_shape, name='inputs')

    # convolutional layers
    conv1, pool1 = convolution_stack_layer(inputs, 16, (2, 2))

    # u-turn
    conv3 = convolution_stack_layer(pool1, 32, (2, 2), polling=False)

    # upsamplig
    up2 = up_block(conv3, conv1, 16)

    # output
    layer = Conv2D(1, (2, 2), padding='same')(up2)
    output = Activation("sigmoid")(layer)

    return Model(inputs, output)


def train_segmentation_model(X, Y, test_x, test_classes, now):
    """
    Train of the model previously created
    Args:
        now:
        test_classes: classes of the teste set
        test_x: imagens of the test set
        X: features for the trains
        Y: classes for the train

    Returns: the masks predicted

    """
    log_dir = "{}/Segmentation{}/".format(root_logdir, now)

    trainX, valX, trainY, valY = train_test_split(X, Y, train_size=3500, test_size=500)
    tb_callback = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, profile_batch=0)
    model = create_segmentation_model((64, 64, 3))

    opt = Adam()

    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=["binary_accuracy"])
    model.summary()

    model.fit(trainX, trainY,
              validation_data=(valX, valY),
              batch_size=32,
              epochs=20,
              callbacks=[tb_callback])

    model.save_weights('pesos/segmentation{}.h5'.format(now))

    # model.load_weights('pesos/segmentation_best_model.h5')

    # measure test error
    segmentation_eval = model.evaluate(test_x, test_classes)
    print(f'Segmentation accuracy: {segmentation_eval[1]}; Segmentation loss: {segmentation_eval[0]}')

    predictions = model.predict(test_x)

    return predictions


# unused model
def convolution_unconvolution_model(input_shape):
    inputs = Input(shape=input_shape, name='inputs')

    # convolutional layers stacked
    convolutional_layer = convolution_stack_layer(inputs, 128, (2, 2))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 64, (2, 2))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 32, (2, 2))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 16, (2, 2))

    # upsamplig
    layer = UpSampling2D(size=(2, 2), interpolation="nearest")(convolutional_layer)
    layer = UpSampling2D(size=(2, 2), interpolation="nearest")(layer)
    layer = UpSampling2D(size=(2, 2), interpolation="nearest")(layer)

    # output
    layer = Conv2D(1, (2, 2), padding='same')(layer)
    output = Activation("sigmoid")(layer)
    return Model(inputs, output)
