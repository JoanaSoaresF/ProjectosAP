# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

root_logdir = "logs"


# now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
# log_dir = "{}/Multiclass-{}/".format(root_logdir, now)


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
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

    return layer


def dense_block(inputs, n, dropout=True):
    """
    Creates a dense block with Relu as activation function and batch normalization
    Args:
        inputs: the previous layer in the network
        n: the number of neurons of the dense layer
        dropout: boolean to specify is a layer of dropout is to be added

    Returns:

    """
    layer = Dense(n)(inputs)
    layer = Activation("relu")(layer)
    # Use batch normalization after activation so that input of following layer is standardized
    layer = BatchNormalization()(layer)
    if dropout:
        layer = Dropout(0.5)(layer)
        # Use dropout for dense layers
    return layer


def create_multiclass_model(input_shape):
    """
    Creates the model defined for the multiclass problem
    Args:
        input_shape: shape of the original input

    Returns: the model created

    """
    inputs = Input(shape=input_shape, name='inputs')

    # convolutional layers stacked
    convolutional_layer = convolution_stack_layer(inputs, 64, (3, 3))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 32, (3, 3))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 16, (3, 3))

    # Flatten before dense layers
    layer = Flatten()(convolutional_layer)

    # Dense layers
    layer = dense_block(layer, 256)
    layer = dense_block(layer, 128)
    layer = dense_block(layer, 64)

    # output
    layer = Dense(10)(layer)
    output = Activation("softmax")(layer)

    return Model(inputs, output)


def train_multiclass_model(X, Y, test_x, test_classes, now):
    """
    Train of the model previously created
    Args:
        now:
        test_classes: classes of the teste set
        test_x: imagens of the test set
        X: features for the trains
        Y: classes for the train

    Returns: the model trained

    """
    log_dir = "{}/Multiclass-{}/".format(root_logdir, now)
    trainX, valX, trainY, valY = train_test_split(X, Y, train_size=3500, test_size=500)
    tb_callback = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, profile_batch=0)
    model = create_multiclass_model((64, 64, 3))

    opt = Adam()

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["categorical_accuracy"])
    model.summary()

    model.fit(trainX, trainY,
              validation_data=(valX, valY),
              batch_size=32,
              epochs=100,
              callbacks=[tb_callback])

    model.save_weights('multiclass_best_model.h5')

    # model.load_weights('pesos/multiclass-best_model.h5')

    # measure test error
    multiclass_eval = model.evaluate(test_x, test_classes)
    print(f'Multiclass accuracy: {multiclass_eval[1]}; Multiclass loss: {multiclass_eval[0]}')

    return model


# unused model
def model_with_residual_block(input_shape):
    inputs = Input(shape=input_shape, name='inputs')
    c1 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(c1)
    c1 = Concatenate()([c1, inputs])
    c1 = Activation("relu")(c1)
    layer = MaxPooling2D(pool_size=(2, 2))(c1)

    # Flatten before dense layers
    layer = Flatten()(layer)

    # Dense layers
    layer = dense_block(layer, 256)
    layer = dense_block(layer, 128)
    layer = dense_block(layer, 64)

    # output
    layer = Dense(10)(layer)
    output = Activation("softmax")(layer)

    return Model(inputs, output)
