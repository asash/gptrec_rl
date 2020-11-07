from aprec.recommenders.metrics.ndcg import KerasNDCG
import tensorflow.keras.backend as K
import numpy as np
import unittest

class TestKerasNDCG(unittest.TestCase):
    def test_keras_ndcg(self):
        y_true = K.constant(np.array([[0, 1, 0], [1, 1, 0]]))
        y_pred = K.constant(np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]]))
        keras_ndcg = KerasNDCG(2)
        res = keras_ndcg(y_true, y_pred)
        K.print_tensor(res)

if __name__ == "__main__":
    unittest.main()
