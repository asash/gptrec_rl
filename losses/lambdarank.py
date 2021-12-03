import tensorflow as tf
import tensorflow.keras.backend as K

def print_tensor(name, tensor):
    print(name)
    K.print_tensor(tensor)


class LambdaRankLoss(object):
    def get_pairwise_diffs_for_vector(self, x):
        a, b = tf.meshgrid(x[:self.ndcg_at], tf.transpose(x))
        result = tf.subtract(b, a)
        return  result

    def get_pairwise_diff_batch(self, x):
        x_top_tile = tf.tile(tf.expand_dims(x[:,:self.ndcg_at], 1), [1, self.n_items, 1])
        x_tile = tf.tile(tf.expand_dims(x, 2), [1, 1, self.ndcg_at])
        result = x_tile - x_top_tile
        return result

    def need_swap_batch(self, x):
        return tf.cast(tf.sign(self.get_pairwise_diff_batch(x)), self.dtype)

    def __init__(self, n_items, batch_size, sigma=1.0, ndcg_at = 30, dtype=tf.float32, lambda_normalization=True):
        self.__name__ = 'lambdarank'
        self.batch_size = batch_size
        self.sigma = sigma
        self.n_items = n_items
        self.ndcg_at = min(ndcg_at, n_items)
        self.dtype = dtype
        self.dcg_position_discounts = 1. / tf.experimental.numpy.log2(tf.cast(tf.range(n_items) + 2, self.dtype))
        self.top_position_discounts = tf.reshape(self.dcg_position_discounts[:self.ndcg_at], (self.ndcg_at, 1))
        self.swap_importance = tf.abs(self.get_pairwise_diffs_for_vector(self.dcg_position_discounts))
        self.batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(self.batch_size), 1), [1, self.n_items]),
                                        (n_items * batch_size, 1))
        self.mask = tf.cast(tf.reshape(1 - tf.pad(tf.ones(self.ndcg_at), [[0, self.n_items - self.ndcg_at]]), (1, self.n_items)), self.dtype)
        self.lambda_normalization = lambda_normalization

    @tf.custom_gradient
    def __call__(self, y_true, y_pred):
        result = tf.reduce_mean(tf.abs(y_pred))
        def grad(dy):
            lambdas = self.get_lambdas(y_true, y_pred)
            return 0 * dy, lambdas * dy
        return result, grad

    @tf.function
    def get_lambdas(self, y_true, y_pred):
        sorted_by_score = tf.nn.top_k(tf.cast(y_pred, self.dtype), self.n_items)
        col_indices_reshaped = tf.reshape(sorted_by_score.indices, (self.n_items * self.batch_size, 1))
        pred_ordered = sorted_by_score.values
        best_score = pred_ordered[:, 0]
        worst_score = pred_ordered[:, -1]

        range_is_zero = tf.reshape(tf.cast(tf.math.equal(best_score, worst_score), self.dtype), (self.batch_size, 1, 1))

        true_ordered = tf.gather(tf.cast(y_true, self.dtype), sorted_by_score.indices, batch_dims=1)
        inverse_idcg = self.get_inverse_idcg(true_ordered)
        S = self.need_swap_batch(true_ordered)
        true_gains = 2 ** true_ordered - 1
        true_gains_diff = self.get_pairwise_diff_batch(true_gains)
        abs_delta_ndcg = tf.abs(true_gains_diff * self.swap_importance) * inverse_idcg
        pairwise_diffs = self.get_pairwise_diff_batch(pred_ordered) * S
        #normalize dcg gaps - inspired by lightbm
        norms = (1 - range_is_zero) * (tf.abs(pairwise_diffs) + 0.01) + (range_is_zero)
        abs_delta_ndcg = tf.math.divide_no_nan(abs_delta_ndcg, norms)


        sigmoid = 1.0 / (1 + tf.exp(self.sigma * (pairwise_diffs)))

        lambda_matrix = -self.sigma * abs_delta_ndcg * sigmoid * S


        #calculate sum of lambdas by rows. For top items - calculate as sum by columns.
        lambda_sum_raw = tf.reduce_sum(lambda_matrix, axis=2)
        top_lambda_sum = tf.pad(-tf.reduce_sum(lambda_matrix, axis=1), [[0, 0], [0, self.n_items - self.ndcg_at]])
        lambda_sum_raw_top_masked = lambda_sum_raw * self.mask
        lambda_sum_result = lambda_sum_raw_top_masked + top_lambda_sum

        if self.lambda_normalization:
            #normalize results - inspired by lightbm
            all_lambdas_sum = tf.reshape(tf.reduce_sum(tf.abs(lambda_sum_result), axis=(1)), (self.batch_size, 1))
            norm_factor = tf.math.divide_no_nan(tf.experimental.numpy.log2(all_lambdas_sum + 1), all_lambdas_sum )

            lambda_sum = lambda_sum_result * norm_factor
        else:
            lambda_sum = lambda_sum_result

        indices = tf.concat([self.batch_indices, col_indices_reshaped], axis=1)

        result_lambdas = tf.scatter_nd(indices, tf.reshape(lambda_sum, [self.n_items * self.batch_size]),
                                       tf.constant([self.batch_size, self.n_items]))
        return tf.cast(result_lambdas, tf.float32)

    def get_inverse_idcg(self, true_ordered):
        top_k_values = tf.nn.top_k(true_ordered, self.ndcg_at).values
        top_k_discounted = tf.linalg.matmul(top_k_values, self.top_position_discounts)
        return tf.reshape(tf.math.divide_no_nan(tf.cast(1.0, self.dtype), top_k_discounted), (self.batch_size, 1, 1))


