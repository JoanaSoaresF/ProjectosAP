# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################

from keras import Input, Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import TensorBoard
from keras.layers import GlobalAveragePooling2D, Activation, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from TP1.multiclass_models import dense_block

root_logdir = "logs"


def create_mobile_net_model_multiclass(input_shape):
    """
    Creates the model defined for the classification multiclass problem using MobileNet to learn the features
    Args:
        input_shape: shape of the original input

    Returns: the model created

    """

    # base model for Mobile Net
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, classes=10)
    base_model.trainable = False

    # New Model
    inputs = Input(shape=input_shape, name='inputs')
    layer = base_model(inputs, training=False)
    layer = GlobalAveragePooling2D()(layer)

    # Dense layers for classification
    layer = dense_block(layer, 512, dropout=True)
    layer = dense_block(layer, 256, dropout=True)
    layer = dense_block(layer, 128, dropout=True)
    layer = dense_block(layer, 64, dropout=True)

    # output
    layer = Dense(10)(layer)
    output = Activation("softmax")(layer)

    return Model(inputs, output)


def train_mobile_net_multiclass_model(X, Y, test_x, test_classes, now):
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
    log_dir = "{}/MobileNet-Multiclass{}/".format(root_logdir, now)

    trainX, valX, trainY, valY = train_test_split(X, Y, train_size=3500, test_size=500)
    tb_callback = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, profile_batch=0)
    model = create_mobile_net_model_multiclass((64, 64, 3))

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

    model.save_weights('pesos/mobile_net_multiclass_model{}.h5'.format(now))

    # model.load_weights('pesos/mobile_net_multiclass_model_2022-05-08_08h31min.h5')

    # measure test error
    eval = model.evaluate(test_x, test_classes)
    print(f'MobileNet Multiclass accuracy: {eval[1]}; MobileNet Multiclass loss: {eval[0]}')

    return model
