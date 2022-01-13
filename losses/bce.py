from aprec.losses.loss import Loss
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


class BCELoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__name__ = "BCE"
        self.less_is_better = True

    def __call__(self, y_true, y_pred):
        EPS = 1e-24
        y_true_float = tf.cast(y_true, 'float32')
        is_target = tf.cast((y_true_float >= -EPS), 'float32')
        trues = y_true_float*is_target
        pos = -trues*tf.math.log((tf.sigmoid(y_pred) + EPS)) * is_target
        neg = -(1.0 - trues)*tf.math.log((1.0 - tf.sigmoid(y_pred)) + EPS) * is_target
        num_targets = tf.reduce_sum(is_target)
        res_sum = tf.math.divide_no_nan(tf.reduce_sum(pos + neg), num_targets)
        return res_sum