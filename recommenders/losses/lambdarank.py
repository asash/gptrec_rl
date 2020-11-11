import math

from tensorflow.keras import backend as K
import tensorflow as tf


def get_pairwise_diff_vector(x):
    a, b = tf.meshgrid(x, tf.transpose(x))
    return tf.subtract(b, a)

def get_pairwise_diff_batch(x):
    return tf.map_fn(get_pairwise_diff_vector, x)

def need_swap_vector(x):
    diffs = tf.cast(tf.sign(get_pairwise_diff_vector(x)), tf.float32)
    return diffs

def need_swap_batch(x):
    return tf.map_fn(need_swap_vector, x)


class PairwiseLoss(object):
    def __init__(self, k, batch_size, sigma=0.5):
        self.__name__ = 'pairwise_loss'
        self.batch_size = batch_size
        self.sigma = sigma
        self.k = k
        self.dcg_position_discounts = tf.math.log(2.0) / tf.math.log(tf.cast(tf.range(k) + 2, tf.float32))
        self.swap_importance = tf.abs(get_pairwise_diff_vector(self.dcg_position_discounts))

    def dcg(self, x):
        return tf.reduce_sum((2 ** x - 1) * self.dcg_position_discounts)

    def inverse_idcg(self, x):
        top_k = tf.nn.top_k(x, self.k)
        return tf.math.divide_no_nan(1., self.dcg(top_k.values))


    @tf.custom_gradient
    def __call__(self, y_true, y_pred):
        result = tf.reduce_mean(tf.abs(y_pred))
        def grad(dy):
            lambdas = self.get_lambdas(y_true, y_pred)
            return 0 * dy, lambdas * dy
        return result, grad

    def get_lambdas(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, self.k)
        col_indices = top_k.indices
        pred_ordered = top_k.values
        best_score = pred_ordered[:, 0]
        worst_score = pred_ordered[:, -1]

        range_is_zero = tf.reshape(tf.cast(tf.math.equal(best_score, worst_score), tf.float32), (self.batch_size, 1, 1))

        true_ordered = tf.gather(y_true, top_k.indices, batch_dims=1)
        inverse_idcg = tf.reshape(tf.map_fn(self.inverse_idcg, true_ordered), (self.batch_size, 1, 1))
        #K.print_tensor(inverse_idcg)
        S = need_swap_batch(true_ordered)
        true_gains = 2 ** true_ordered - 1
        true_gains_diff = get_pairwise_diff_batch(true_gains)
        abs_delta_ndcg = tf.abs(true_gains_diff * self.swap_importance) * inverse_idcg
        #K.print_tensor(self.swap_importance)
        pairwise_diffs = get_pairwise_diff_batch(pred_ordered) * S

        norms = (1 - range_is_zero) * (tf.abs(pairwise_diffs) + 0.01) + (range_is_zero)
        abs_delta_ndcg = abs_delta_ndcg / norms
        sigmoid = 1.0 / (1 + tf.exp(self.sigma * (pairwise_diffs)))
        #ranknet lambdas
        #lambda_matrix = self.sigma * (0.5 * (1 - S) -sigmoid)
        lambda_matrix = -self.sigma * abs_delta_ndcg * sigmoid * S
        all_lambdas_sum = tf.reduce_sum(tf.abs(lambda_matrix))

        norm_factor = tf.math.log(all_lambdas_sum + 1) / (all_lambdas_sum * tf.math.log(2.0))
        lambda_sum = tf.reduce_sum(lambda_matrix, axis=2) * norm_factor

        #K.print_tensor(lambda_sum)
        batch_indices = tf.reshape(tf.repeat(tf.range(self.batch_size), self.k), (self.batch_size, self.k))
        indices = tf.reshape(tf.stack([batch_indices, col_indices], axis=2), (self.k*self.batch_size, 2))
        result_lambdas = tf.scatter_nd(indices, tf.reshape(lambda_sum,[self.k*self.batch_size]),
                                       tf.constant([self.batch_size, self.k]))
        return result_lambdas


class LambdaRankLoss(object):
    def __init__(self, k=20, eps=1e-6):
        self.eps = eps
        self.k = k
        self.__name__ = f"lambdarank_{k}"
        discount = []
        for i in range(1, k + 1):
            discount.append(1 / math.log2(i + 1))
        self.discount = K.constant(discount)

    def __call__(self, y_true, y_pred):
        z = get_pairwise_diff_batch(y_pred)
        #K.print_tensor(z)
        return self.get_res_pred_discounted(y_true, y_pred) + self.get_res_true_discounted(y_true, y_pred)

    def get_res_pred_discounted(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, self.k)
        true_vals = tf.gather(y_true, top_k.indices, batch_dims=1)
        positive_log = K.log(top_k.values + self.eps)
        negative_log = K.log(1 - top_k.values + self.eps)
        return K.mean((-true_vals * positive_log - (1 - true_vals) * negative_log) * self.discount)

    def get_res_true_discounted(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_true, self.k)
        chired_vals = tf.gather(y_pred, top_k.indices, batch_dims=1)
        positive_log = K.log(pred_vals + self.eps)
        negative_log = K.log(1 - pred_vals + self.eps)
        return K.mean((-top_k.values * positive_log - (1 - top_k.values) * negative_log) * self.discount)
