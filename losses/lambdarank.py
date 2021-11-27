import tensorflow as tf
import tensorflow.keras.backend as K

def print_tensor(name, tensor):
    print(name)
    K.print_tensor(tensor)

class LambdaRankLoss(object):
    def get_pairwise_diffs_for_vector(self, x):
        a, b = tf.meshgrid(x[:self.ndcg_at], tf.transpose(x))
        return tf.subtract(b, a)

    def get_pairwise_diff_batch(self, x):
        return tf.vectorized_map(self.get_pairwise_diffs_for_vector, x)

    def need_swap_vector(self, x):
        return tf.cast(tf.sign(self.get_pairwise_diffs_for_vector(x)), tf.float32)

    def need_swap_batch(self, x):
        return tf.vectorized_map(self.need_swap_vector, x)

    def __init__(self, n_items, batch_size, sigma=1.0, ndcg_at = 30):
        self.__name__ = 'lambdarank'
        self.batch_size = batch_size
        self.sigma = sigma
        self.n_items = n_items
        self.ndcg_at = min(ndcg_at, n_items)
        self.dcg_position_discounts = tf.math.log(2.0) / tf.math.log(tf.cast(tf.range(n_items) + 2, tf.float32))
        self.top_position_discounts = tf.reshape(self.dcg_position_discounts[:self.ndcg_at], (self.ndcg_at, 1))
        self.swap_importance = tf.abs(self.get_pairwise_diffs_for_vector(self.dcg_position_discounts))

    def dcg(self, x):
        return tf.reduce_sum((2 ** x - 1) * self.dcg_position_discounts[:x.shape[0]])

    def inverse_idcg(self, x):
        top_k = tf.nn.top_k(x, self.ndcg_at)
        return tf.math.divide_no_nan(1., self.dcg(top_k.values))


    @tf.custom_gradient
    def __call__(self, y_true, y_pred):
        result = tf.reduce_mean(tf.abs(y_pred))
        def grad(dy):
            lambdas = self.get_lambdas(y_true, y_pred)
            return 0 * dy, lambdas * dy
        return result, grad


    @tf.function
    def get_lambdas(self, y_true, y_pred):
        sorted_by_score = tf.nn.top_k(y_pred, self.n_items)
        col_indices = sorted_by_score.indices
        pred_ordered = sorted_by_score.values
        best_score = pred_ordered[:, 0]
        worst_score = pred_ordered[:, -1]

        range_is_zero = tf.reshape(tf.cast(tf.math.equal(best_score, worst_score), tf.float32), (self.batch_size, 1, 1))

        true_ordered = tf.gather(y_true, sorted_by_score.indices, batch_dims=1)
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

        #ranknet lambdas
        #lambda_matrix = self.sigma * (0.5 * (1 - S) -sigmoid)
        lambda_matrix = -self.sigma * abs_delta_ndcg * sigmoid * S


        #calculate sum of lambdas by rows. For top items - calculate as sum by columns.
        lambda_sum_raw = tf.reduce_sum(lambda_matrix, axis=2)
        zeros = tf.zeros((self.batch_size, self.n_items - self.ndcg_at, 1))
        top_lambda_sum = tf.pad(-tf.reduce_sum(lambda_matrix, axis=1), [[0, 0], [0, self.n_items - self.ndcg_at]])
        mask = tf.reshape(1 - tf.pad(tf.ones(self.ndcg_at), [[0, self.n_items - self.ndcg_at]]), (1, self.n_items))
        lambda_sum_raw_top_masked = lambda_sum_raw * mask
        lambda_sum_result = lambda_sum_raw_top_masked + top_lambda_sum

        #normalize results - inspired by lightbm
        all_lambdas_sum = tf.reshape(tf.reduce_sum(tf.abs(lambda_sum_result), axis=(1)), (self.batch_size, 1))
        norm_factor = tf.math.divide_no_nan(tf.math.log(all_lambdas_sum + 1), (all_lambdas_sum * tf.math.log(2.0)))
        lambda_sum = lambda_sum_result * norm_factor

        batch_indices = tf.reshape(tf.repeat(tf.range(self.batch_size), self.n_items), (self.batch_size, self.n_items))
        indices = tf.reshape(tf.stack([batch_indices, col_indices], axis=2), (self.n_items * self.batch_size, 2))
        result_lambdas = tf.scatter_nd(indices, tf.reshape(lambda_sum, [self.n_items * self.batch_size]),
                                       tf.constant([self.batch_size, self.n_items]))
        return result_lambdas

    def get_inverse_idcg(self, true_ordered):
        top_k_values = tf.nn.top_k(true_ordered, self.ndcg_at).values
        top_k_discounted = tf.linalg.matmul(top_k_values, self.top_position_discounts)
        result = tf.reshape(tf.math.divide_no_nan(1.0, top_k_discounted), (self.batch_size, 1, 1))
        return result


