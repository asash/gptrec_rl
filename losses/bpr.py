import tensorflow as tf

from aprec.losses.loss_utils import get_pairwise_diff_batch, get_truncated
from aprec.losses.loss import Loss


class BPRLoss(Loss):
    def __init__(self, num_items=None, batch_size=None, max_positives=10, pred_truncate=None):
        super().__init__(num_items, batch_size)
        self.max_positives = max_positives
        self.pred_truncate = pred_truncate

    def __call__(self, y_true, y_pred):
        top_true = tf.math.top_k(y_true, self.max_positives)
        pred_ordered_by_true = tf.gather(y_pred, top_true.indices, batch_dims=1)

        pred, true_ordered_by_pred = get_truncated(y_true, y_pred, self.pred_truncate) 
        pred_size = tf.shape(pred)[-1]

        mask = tf.cast((get_pairwise_diff_batch(top_true.values, true_ordered_by_pred, self.max_positives, pred_size) > 0), tf.float32)
        values = get_pairwise_diff_batch(pred_ordered_by_true, pred, self.max_positives, pred_size)
        sigmoid =  -tf.math.log_sigmoid(values) * mask
        result = tf.reduce_sum(sigmoid, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
        return tf.reduce_mean(result)

