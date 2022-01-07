from aprec.losses.loss import Loss
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy


class BCELoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__name__ = "BCE"
        self.less_is_better = True

    def __call__(self, y_true_raw, y_pred):
        EPS = 1e-24
        y_true = tf.cast(y_true_raw, 'float32') 
        is_target = tf.cast((y_true >= -EPS), 'float32')
        trues = y_true*is_target
        vals = K.binary_crossentropy(trues, y_pred, from_logits=True) 
        num_targets = tf.reduce_sum(is_target)
        res_sum = tf.reduce_sum(vals)
        return tf.math.divide_no_nan(res_sum, num_targets)