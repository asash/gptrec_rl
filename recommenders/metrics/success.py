import tensorflow as tf
import tensorflow.keras.backend as K

class KerasSuccess(object):
   def __init__(self, k):
        self.k = k
        self.__name__ = f"Success_at_{k}"

   def __call__(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, self.k)
        gains = tf.gather(y_true, top_k.indices, batch_dims=1)
        return  K.mean(gains)

