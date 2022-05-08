#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##############################################################################
#  Aprendizagem Profunda, TP1 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
from datetime import datetime

import tp1_utils as util
from TP1.mobile_net_model_multiclass import train_mobile_net_multiclass_model
from TP1.mobile_net_model_multilabel import train_mobile_net_multilabel_model
from TP1.multilabel_models import train_multilabel_model
from TP1.segmentation_model import train_segmentation_model
from multiclass_models import train_multiclass_model

PROBLEM_MULTICLASS = 0
PROBLEM_MULTILABEL = 1
PROBLEM_SEGMENTATION = 2
MOBILE_NET_PROBLEM_MULTICLASS = 3
MOBILE_NET_PROBLEM_MULTILABEL = 4


# tensorboard --logdir logs
# http://127.0.0.1:6006

def normalization(X):
    X = X.reshape((X.shape[0], 64, 64, 3))
    X = X.astype("float32") / 255.0
    return X


def normalization_mask(X):
    X = X.reshape((X.shape[0], 64, 64, 1))
    X = X.astype("float32") / 255.0
    return X


def main(problem):
    now = datetime.utcnow().strftime("_%Y-%m-%d_%Hh%Mmin")
    data = util.load_data()
    train_X = data['train_X']
    train_classes = data['train_classes']
    train_labels = data['train_labels']
    train_masks = data['train_masks']

    test_x = data['test_X']
    test_classes = data['test_classes']
    test_labels = data['test_labels']
    test_masks = data['test_masks']

    if problem == PROBLEM_MULTICLASS:
        train_multiclass_model(normalization(train_X),
                               train_classes,
                               normalization(test_x),
                               test_classes, now)
    elif problem == PROBLEM_MULTILABEL:
        train_multilabel_model(normalization(train_X),
                               train_labels,
                               normalization(test_x),
                               test_labels, now,
                               load_weights=False)
    elif problem == PROBLEM_SEGMENTATION:
        masks_predictions = train_segmentation_model(train_X, train_masks, test_x, test_masks, now)
        util.compare_masks('images/test_compare{}.png'.format(now), test_masks, masks_predictions)
        util.overlay_masks('images/test_overlay{}.png'.format(now), test_x, masks_predictions)
    elif problem == MOBILE_NET_PROBLEM_MULTICLASS:
        train_mobile_net_multiclass_model(train_X,
                                          train_classes,
                                          test_x,
                                          test_classes, now)
    elif problem == MOBILE_NET_PROBLEM_MULTILABEL:
        train_mobile_net_multilabel_model(train_X,
                                          train_labels,
                                          test_x,
                                          test_labels, now)


main(MOBILE_NET_PROBLEM_MULTICLASS)
