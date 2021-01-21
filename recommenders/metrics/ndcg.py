import math

import tensorflow as tf
import tensorflow.keras.backend as K

class KerasNDCG(object):
    def __init__(self, k):
        self.k = k
        discounts = []
        for i in range(1, k+1):
            discounts.append(1 / math.log2(i + 1))
        self.discounts = K.constant(discounts)
        self.__name__ = f"ndcg_at_{k}"

    def dcg(self, scores):
       gain = K.pow(2.0, scores) - 1
       return K.sum(gain * self.discounts, axis=-1)

    def __call__(self, y_true, y_pred):
        eps = 0.000001
        top_k = tf.nn.top_k(y_pred, self.k)
        gains = tf.gather(y_true, top_k.indices, batch_dims=1)
        dcg_val = self.dcg(gains)

        ideal_top_k = tf.nn.top_k(y_true, self.k)
        ideal_gains = tf.gather(y_true, ideal_top_k.indices, batch_dims=1)
        idcg_val = self.dcg(ideal_gains)
        return K.mean(dcg_val / (idcg_val + eps))

