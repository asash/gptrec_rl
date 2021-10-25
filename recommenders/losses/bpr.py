import tensorflow as tf


# https://stackoverflow.com/questions/37086098/does-tensorflow-map-fn-support-taking-more-than-one-tensor
def my_map(fn, arrays, dtype=tf.float32):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out


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
        values = self.get_pairwise_diffs_matrices(pred_ordered, y_pred)*mask
        result = tf.reduce_sum(tf.math.log_sigmoid(values), axis=1)
        pass
