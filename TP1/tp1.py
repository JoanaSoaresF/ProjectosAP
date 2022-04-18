#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from multiclass_models import train_multiclass_model
import tp1_utils as util


# tensorboard --logdir logs
# http://127.0.0.1:6006

def normalization(X):
    X = X.reshape((X.shape[0], 64, 64, 3))
    X = X.astype("float32") / 255.0
    return X


def main():
    data = util.load_data()
    train_X = normalization(data['train_X'])
    train_classes = data['train_classes']
    testX = normalization(data['test_X'])
    test_classes = data['test_classes']
    model = train_multiclass_model(train_X, train_classes)
    multiclass_eval = model.evaluate(testX, test_classes)
    print(f'Multiclass accuracy: {multiclass_eval[1]}; Multiclass loss: {multiclass_eval[0]}')
    print(multiclass_eval)
    """
    Multiclass accuracy: 0.9879999756813049; Multiclass loss: 0.035707972943782806
    [0.035707972943782806, 0.9879999756813049]
    """


main()
