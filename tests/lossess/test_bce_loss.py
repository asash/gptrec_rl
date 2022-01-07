from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
from aprec.losses.bce import BCELoss
import unittest

class TestBCELoss(unittest.TestCase):
    def test_bce_loss(self):
        y_true = tf.constant([1, 0, 1, 0])
        y_pred = [0.1, 0.2, 0.3, 0.4]
        loss = float(BCELoss()(y_true, y_pred))
        keras_loss = float(BinaryCrossentropy(from_logits=True)(y_true, y_pred))
        self.assertAlmostEqual(loss, keras_loss)

        y_true = tf.constant([1., 0, 1, 0])
        y_pred = [-50.0, -50, -50, -50]
        loss = float(BCELoss()(y_true, y_pred))
        keras_loss = float(BinaryCrossentropy(from_logits=True)(y_true, y_pred))
        self.assertAlmostEqual(loss, keras_loss)

        y_true = tf.constant([-1., -1, -1, -1])
        y_pred = [-50.0, -50, -50, -50]
        loss = float(BCELoss()(y_true, y_pred))
        self.assertAlmostEqual(loss, 0.0)





