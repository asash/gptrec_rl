import unittest
import tensorflow.keras.backend as K

from aprec.recommenders.losses.bpr import BPRLoss


class TestBPRLoss(unittest.TestCase):
        def test_bpr_loss(self):
            bpr_loss = BPRLoss(3)
            val = bpr_loss(K.constant([[0, 0, 1, 1],
                                 [0, 0, 1, 1]]),
                     K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1]]))
            self.assertAlmostEqual(val, 0.56441593)

            poor_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[1, 0.5, 0, 0]]))
            avg_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0.1, 0.3, 1, 0]]))
            good_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0, 0, 1, 1]]))
            assert (poor_pred_loss > avg_pred_loss)
            assert (good_pred_loss < avg_pred_loss)
