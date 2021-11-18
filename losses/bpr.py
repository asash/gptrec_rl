import tensorflow as tf

from aprec.losses.loss_utils import my_map


class BPRLoss(object):
    def __init__(self, max_positives=10):
        self.max_positives = max_positives


    def get_pairwise_diffs_matrix(self, x, y):
        a, b = tf.meshgrid(tf.transpose(y), x)
        return tf.subtract(b, a)

    def get_pairwise_diffs_matrices(self, a, b):
        result = my_map(self.get_pairwise_diffs_matrix, (a, b))
        return result

    def __call__(self, y_true, y_pred):
        top_true = tf.math.top_k(y_true, self.max_positives)
        pred_ordered = tf.gather(y_pred, top_true.indices, batch_dims=1)
        mask = tf.cast((self.get_pairwise_diffs_matrices(top_true.values, y_true) > 0), tf.float32)
        values = self.get_pairwise_diffs_matrices(pred_ordered, y_pred)
        sigmoid =  -tf.math.log_sigmoid(values) * mask
        result = tf.reduce_sum(sigmoid, axis=[1, 2])
        return tf.reduce_mean(result)