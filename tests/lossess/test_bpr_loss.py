import unittest
import math
import tensorflow as tf
import random
import tensorflow.keras.backend as K

from aprec.losses.bpr import BPRLoss


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def naive_bpr_impl(y_true, y_pred):
    n_pairs = 0
    loss = 0.0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if y_true[i] > 0.5 and y_true[j] < 0.5:
                n_pairs += 1
                positive = y_pred[i]
                negative = y_pred[j]
                diff = positive - negative
                sigm = sigmoid(diff)
                loss -= math.log(sigm)
    return loss/n_pairs


    

class TestBPRLoss(unittest.TestCase):
        def compare_with_naive(self, a, b):
            bpr_loss = BPRLoss(max_positives=len(a))
            naive_bpr_los_val = naive_bpr_impl(a, b)
            computed_loss_val = float(bpr_loss(tf.constant([a]), tf.constant([b])))
            self.assertAlmostEquals(computed_loss_val, naive_bpr_los_val, places=4)
            
        def test_compare_with_naive(self):
                self.compare_with_naive([0.0, 1.], [0.1, 0])
                random.seed(31337)
                for i in range(100):
                    sample_len = random.randint(2, 1000)
                    y_true = []
                    y_pred = []
                    for j in range(sample_len):
                        y_true.append(random.randint(0, 1) * 1.0)
                        y_pred.append(random.random())
                    self.compare_with_naive(y_true, y_pred)

        def test_bpr_loss(self):
            bpr_loss = BPRLoss(max_positives=3)
            
            val = bpr_loss(K.constant([[0, 0, 1, 1],
                                 [0, 0, 1, 1]]),
                     K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1]]))
            self.assertAlmostEqual(float(val), 0.22475868463516235, places=4)
            poor_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[1, 0.5, 0, 0]]))
            avg_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0.1, 0.3, 1, 0]]))
            good_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0, 0, 1, 1]]))
            self.assertGreater (poor_pred_loss, avg_pred_loss)
            self.assertLess (good_pred_loss, avg_pred_loss)


if __name__ == "__main__":
    unittest.main()
