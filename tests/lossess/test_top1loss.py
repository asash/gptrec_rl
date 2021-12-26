import unittest
import math
import tensorflow as tf
import random
import tensorflow.keras.backend as K

from aprec.losses.top1 import TOP1Loss


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def naive_top1_impl(y_true, y_pred):
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
                loss += sigm + negative ** 2 
    return loss/n_pairs


    

class TestBPRLoss(unittest.TestCase):
        def compare_with_naive(self, a, b, ordered=False):
            if not ordered:
                top1_loss = TOP1Loss()
            else:
                top1_loss = TOP1Loss(pred_truncate=len(a))
            naive_bpr_los_val = naive_top1_impl(a, b)
            computed_loss_val = float(top1_loss(tf.constant([a]), tf.constant([b])))
            self.assertAlmostEquals(computed_loss_val, naive_bpr_los_val, places=4)
            
        def test_compare_with_naive(self):
                self.compare_with_naive([0.0, 1.], [0.2, 0.1])
                random.seed(31337)
                for i in range(100):
                    ordered = bool(random.randint(0, 1))
                    sample_len = random.randint(2, 5)
                    y_pred = []
                    y_true = [0] * sample_len
                    y_true[random.randint(0, sample_len - 1)] = 1.0
                    for j in range(sample_len):
                        y_pred.append(random.random())
                    self.compare_with_naive(y_true, y_pred, ordered)

        def test_top1_loss(self):
            top1_loss = TOP1Loss() 
            val = top1_loss(K.constant([[0, 0, 0, 1.0],
                                 [0, 0, 1., 0]]),
                     K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1.0]]))
            self.assertAlmostEqual(float(val), 0.8719395399093628, places=4)

        def test_top1_truncate(self):
            top1_loss = TOP1Loss(pred_truncate=1) 
            val = float(top1_loss(tf.constant([[0, 0, 0, 1]]), tf.constant([[0.1, 0.3, 0, 0]])))
            self.assertAlmostEqual(val, 0.515557483188341)
 


if __name__ == "__main__":
   unittest.main()