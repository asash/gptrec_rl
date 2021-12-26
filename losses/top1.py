import tensorflow as tf
from aprec.losses.loss_utils import get_pairwise_diff_batch, get_truncated

from aprec.losses.loss import Loss

#TOP1 loss as defined in GRU4rec Papper https://arxiv.org/pdf/1511.06939
#We assume that there is only one positive sample. 
#If there are more then one posive, the one will be sampled randomly. 
class TOP1Loss(Loss):
    def __init__(self, num_items=None, batch_size=None, pred_truncate=None):
        super().__init__(num_items, batch_size)
        self.pred_truncate = pred_truncate

    def __call__(self, y_true, y_pred):
        top_true = tf.math.top_k(y_true)
        positive_true = top_true.values
        positive_pred = tf.gather(y_pred, top_true.indices, batch_dims=1)
        pred, true_ordered_by_pred = get_truncated(y_true, y_pred, self.pred_truncate) 
        diff = positive_pred - pred        
        mask = tf.cast(true_ordered_by_pred < positive_true, 'float32')
        sigm = tf.sigmoid(diff) * mask
        square = pred * pred * mask
        result = tf.reduce_sum(sigm + square, axis=[1] ) / tf.reduce_sum(mask, axis=1)
        return tf.reduce_mean(result)
