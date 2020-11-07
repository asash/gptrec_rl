import math

from tensorflow.keras import backend as K
import tensorflow as tf


def get_pairwise_diff_vector(x):
    a, b = tf.meshgrid(x, tf.transpose(x))
    return tf.subtract(b, a)


def need_swap_vector(x):
    diffs = tf.linalg.band_part(get_pairwise_diff_vector(x), 0, -1)
    return tf.dtypes.cast(tf.math.less(diffs, 0), tf.float32)

def get_pairwise_diff_batch(x):
    return tf.map_fn(get_pairwise_diff_vector, x)

def need_swap_batch(x):
    return tf.map_fn(need_swap_vector, x)


@tf.custom_gradient
pairwise_loss()

class LambdaRankLoss(object):
    def __init__(self, k=20, eps=1e-6):
        self.eps = eps 
        self.k = k
        self.__name__ = f"lambdarank_{k}"
        discount = []
        for i in range(1, k + 1):
            discount.append(1/math.log2(i + 1))
        self.discount = K.constant(discount)

    def __call__(self, y_true, y_pred):
        z = get_pairwise_diff_batch(y_pred)
        K.print_tensor(z)
        return self.get_res_pred_discounted(y_true, y_pred) + self.get_res_true_discounted(y_true, y_pred)

    def get_res_pred_discounted(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, self.k)
        true_vals = tf.gather(y_true, top_k.indices, batch_dims=1)
        positive_log = K.log(top_k.values + self.eps)
        negative_log = K.log(1-top_k.values + self.eps)
        return K.mean((-true_vals*positive_log - (1 - true_vals) * negative_log) * self.discount )

    def get_res_true_discounted(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_true, self.k)
        pred_vals = tf.gather(y_pred, top_k.indices, batch_dims=1)
        positive_log = K.log(pred_vals + self.eps)
        negative_log = K.log(1-pred_vals + self.eps)
        return K.mean((-top_k.values*positive_log - (1 - top_k.values) * negative_log) * self.discount)




