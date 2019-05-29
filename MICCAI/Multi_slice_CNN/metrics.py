from sklearn.metrics import confusion_matrix, roc_auc_score
from keras import backend as K
import numpy as np

def auc(y_true, y_pred):
    """Compute Area Under ROC Curve for model predictions"""
    return roc_auc_score(np.asarray(y_true), np.asarray(y_pred))


def conf(y_true, y_pred):
    """Compute accuracy, sensitivity, specificity from model predictions"""
    conf = confusion_matrix(y_true, np.round(y_pred))
    print(conf)
    acc = (conf[0,0]+conf[1,1])/float(np.sum(conf))
    sens = conf[1,1]/(conf[1,1]+conf[1,0])
    spec = conf[0,0]/(conf[0,0]+conf[0,1])
    return acc, sens, spec
    
