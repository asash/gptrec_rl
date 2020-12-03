import tensorflow as tf
class XENDCGLoss(object):
    def __init__(self, n_items, batch_size):
        self.__name__ = 'xendcg'
        self.batch_size = batch_size
        self.n_items = n_items

    def __call__(self, true, pred):
        eps = 1e-5
        gamma = tf.random.uniform(shape=(self.batch_size, self.n_items))
        true_transformed = (2 ** true) - gamma
        true_transformed_sum = tf.expand_dims(tf.math.reduce_sum(true_transformed, axis=1),1)
        true_probs = true_transformed / (true_transformed_sum + eps)

        pred_transformed = tf.exp(pred)
        pred_transformed_sum = tf.expand_dims(tf.math.reduce_sum(pred_transformed, axis=1),1)
        pred_probs = pred_transformed / (pred_transformed_sum + eps)

        result = -tf.math.reduce_sum(true_probs * tf.math.log(pred_probs), axis=1)
        return result
