from aprec.recommenders.losses.lambdarank import LambdaRankLoss, get_pairwise_diff_batch, need_swap_vector, PairwiseLoss
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import numpy as np
import unittest

from tensorflow.python.keras.optimizers import SGD


class TestLambdarankLoss(unittest.TestCase):
   def test_get_lambdas(self):
       batch_size = 1
       n_items = 10000 
       loss = PairwiseLoss(n_items, batch_size,  1)
       for i in range(100):
           print(i)
           y_true = tf.random.uniform((batch_size, n_items))
           y_pred =  tf.random.uniform((batch_size, n_items))
           loss.get_lambdas(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()
