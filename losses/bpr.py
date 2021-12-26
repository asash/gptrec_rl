import tensorflow as tf

from aprec.losses.loss_utils import get_pairwise_diff_batch, my_map
from aprec.losses.loss import Loss


class BPRLoss(Loss):
    def __init__(self, num_items=None, batch_size=None, max_positives=10):
        super().__init__(num_items, batch_size)
        self.max_positives = max_positives

    def __call__(self, y_true, y_pred):
        top_true = tf.math.top_k(y_true, self.max_positives)
        pred_ordered = tf.gather(y_pred, top_true.indices, batch_dims=1)
        mask = tf.cast((get_pairwise_diff_batch(top_true.values, y_true) > 0), tf.float32)
        values = get_pairwise_diff_batch(pred_ordered, y_pred)
        sigmoid =  -tf.math.log_sigmoid(values) * mask
        result = tf.reduce_sum(sigmoid, axis=[1, 2]) / tf.reduce_sum(mask)
        return tf.reduce_mean(result)
