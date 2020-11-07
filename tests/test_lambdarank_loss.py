from aprec.recommenders.losses.lambdarank import LambdaRankLoss, get_pairwise_diff_batch, need_swap_batch, \
    need_swap_vector
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import numpy as np
import unittest

class TestLambdarankLoss(unittest.TestCase):
    def test_lambdarank_loss(self):
        return
        y_true = K.constant(np.array([[0, 1, 0], [1, 1, 0]]))
        y_pred = K.constant(np.array([[0.1, 0.3, 0.2], [0.4, 0.5, 0.6]]))
        loss = LambdaRankLoss(3)
        loss_val = loss(y_true, y_pred)
        ce_loss = binary_crossentropy(y_true, y_pred)
        #K.print_tensor(ce_loss)
        #K.print_tensor(loss_val)

    def test_pairwise_diff(self):
        loss = LambdaRankLoss(1000)
        x = K.constant([[1, 2, 3], [1, 1, 1]])
        z = get_pairwise_diff_batch(x)
        K.print_tensor(z)

    def test_pairwise_diff(self):
        x = K.constant([[1, 2, 3], [1, 1, 1], [0.1, 0.2, 0.3]])
        z = get_pairwise_diff_batch(x)
        K.print_tensor(z)

    def test_need_swap(self):
        x = K.constant([1, 0, 0.5, 1])
        z = need_swap_vector(x)
        K.print_tensor(z)


if __name__ == "__main__":
    unittest.main()
