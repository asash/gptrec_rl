import keras.losses

from aprec.losses.lambdarank import LambdaRankLoss
import tensorflow as tf
from tqdm import tqdm
import unittest

class TestLambdarankLoss(unittest.TestCase):
   def test_get_lambdas(self):
       batch_size = 128
       n_items = 27278
       loss = LambdaRankLoss(n_items, batch_size, 1)
       for i in tqdm(range(1000)):
           y_true = tf.random.uniform((batch_size, n_items))
           y_pred =  tf.random.uniform((batch_size, n_items))
           #keras.losses.binary_crossentropy(y_true, y_pred)
           loss.get_lambdas(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()
