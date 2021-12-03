import numpy as np

from aprec.losses.lambdarank import LambdaRankLoss
import tensorflow as tf
from tqdm import tqdm
import unittest

class TestLambdaranTime(unittest.TestCase):
   def test_get_lambdas(self):
       batch_size = 128
       positives_per_sample = 5
       n_items = 27278

       y_true = np.zeros((batch_size, n_items))
       cur = 1.0
       for sample_num in range(batch_size):
            positives = np.random.choice((range(n_items)), positives_per_sample, replace=False)
            for positive in positives:
                y_true[sample_num][positive] = cur
                cur *= 0.8
       y_true = tf.constant(y_true)

       loss = LambdaRankLoss(n_items, batch_size, 1, ndcg_at=40, dtype=tf.float32)
       for i in tqdm(range(1000)):
           y_pred =  tf.random.uniform((batch_size, n_items))
           #keras.losses.binary_crossentropy(y_true, y_pred)
           loss.get_lambdas(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()
