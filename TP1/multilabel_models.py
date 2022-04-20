# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
from datetime import datetime

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Activation, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from TP1.multiclass_models import convolution_stack_layer, dense_block

root_logdir = "logs"
pesos_dir = "pesos"
# log_dir = "{}/Multilabel-{}/".format(root_logdir, now)


def create_multilabel_model(input_shape):
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
    output = Activation("sigmoid")(layer)

    return Model(inputs, output)


def train_multilabel_model(X, Y, test_x, test_labels, now, load_weights):
    """
    Train of the model previously created
    Args:
        load_weights: boolean to indicate if it is to load weights from previous moddel
        now:
        X: features for the trains
        Y: classes for the train
        test_labels: labels of the test set
        test_x: imagens of the test set

    Returns: the model trained

    """
    log_dir = "{}/Multilabel-{}/".format(root_logdir, now)

    trainX, valX, trainY, valY = train_test_split(X, Y, train_size=3500, test_size=500)
    tb_callback = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, profile_batch=0)
    model = create_multilabel_model((64, 64, 3))

    opt = Adam()

    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=["binary_accuracy"])
    model.summary()

    if load_weights:
        model.load_weights('pesos/multiclass-best_model.h5')
        multilabel_eval = model.evaluate(X, Y)
        print(f'Multilabel accuracy: {multilabel_eval[1]}; Multilabel loss: {multilabel_eval[0]}')
    else:
        model.fit(trainX, trainY,
                  validation_data=(valX, valY),
                  batch_size=32,
                  epochs=100,
                  callbacks=[tb_callback])
        model.save_weights("{}/Multilabel-{}/".format(pesos_dir, now))

    multilabel_eval = model.evaluate(test_x, test_labels)
    print(f'Multilabel accuracy: {multilabel_eval[1]}; Multilabel loss: {multilabel_eval[0]}')

    return model
