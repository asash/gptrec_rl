import tensorflow as tf
def bpr_loss(y_true, y_pred):
    """
    Inspired by https://github.com/sh0416/bpr/blob/master/train.py
    :param y_true: binary flag; 1 if an item is relevant 0 otherwise
    :param y_pred: predicted item probabilities
    """
    positive_preds = tf.reduce_sum((y_pred * y_true), axis=1)
    negative_preds = tf.reduce_sum(y_pred * (1 - y_true), axis=1)
    diff = positive_preds - negative_preds
    log_prob = tf.math.log_sigmoid(diff)
    return -tf.reduce_sum(log_prob)