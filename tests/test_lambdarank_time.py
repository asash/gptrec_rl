from aprec.recommenders.losses.lambdarank import LambdaRankLoss
import tensorflow as tf
import unittest

class TestLambdarankLoss(unittest.TestCase):
   def test_get_lambdas(self):
       batch_size = 1
       n_items = 10000 
       loss = LambdaRankLoss(n_items, batch_size, 1)
       for i in range(20):
           y_true = tf.random.uniform((batch_size, n_items))
           y_pred =  tf.random.uniform((batch_size, n_items))
           loss.get_lambdas(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()
