from keras import backend as K
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred[:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + 
        K.sum(y_pred_f) + K.epsilon())
