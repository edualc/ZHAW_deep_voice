
# from scipy.cluster.hierarchy import fcluster, linkage
# from scipy.spatial.distance import cdist
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

import numpy as np

def equal_error_rate(y_true, y_pred):
    """
    :param y_true: Ground truth speakers per utterance
    :param y_pred: Predicted speakers per utterance

    :return: The Equal Error Rate (EER)
    """

    #true_scores = np.array(y_true)
    true_scores = np.ones(len(y_true)//2).astype(int)
    scores = np.equal(y_pred[:y_pred.size//2], y_pred[y_pred.size//2:]).astype(int)

    fpr, tpr, thresholds = roc_curve(true_scores, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    print("EER: {}\tThresh: {}".format(eer, thresh))

    return eer
