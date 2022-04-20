#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################

import tp1_utils as util
from TP1.multilabel_models import train_multilabel_model
from multiclass_models import train_multiclass_model

PROBLEM_MULTICLASS = 0
PROBLEM_MULTILABEL = 1
PROBLEM_SEGMENTATION = 2


# tensorboard --logdir logs
# http://127.0.0.1:6006

def normalization(X):
    X = X.reshape((X.shape[0], 64, 64, 3))
    X = X.astype("float32") / 255.0
    return X


def main(problem):
    data = util.load_data()
    train_X = normalization(data['train_X'])
    train_classes = data['train_classes']
    train_labels = data['train_labels']
    testX = normalization(data['test_X'])
    test_classes = data['test_classes']
    test_labels = data['test_labels']
    if problem == PROBLEM_MULTICLASS:
        train_multiclass_model(train_X, train_classes, testX, test_classes)
    elif problem == PROBLEM_MULTILABEL:
        train_multilabel_model(train_X, train_labels, testX, test_labels)


main(PROBLEM_MULTILABEL)
