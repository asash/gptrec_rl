from aprec.recommenders.losses.lambdarank import LambdaRankLoss, get_pairwise_diff_batch, need_swap_vector, PairwiseLoss
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import unittest

from tensorflow.python.keras.optimizers import SGD


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

    def test_get_lambdas(self):
        y_true = K.constant([[1, 0.5, 0, 0]])
        y_pred = K.constant([[0, 0.1, 0.2, 0.3]])
        loss = PairwiseLoss(4, 1, 0.5)
        current = y_pred
        lr = 0.1
        for i in range(200):
            lambdas = loss.get_lambdas(y_true, current)
            current =  current - lr * lambdas
            print(f"iter: {i}")
            print("CURRENT")
            K.print_tensor(current)
            print("LAMBDA")
            K.print_tensor(lambdas)
            print("XXX")

    def test_dcg(self):
        loss = PairwiseLoss(4, 2)
        res = loss.idcg(K.constant([0, 0, 0, 1]))
        K.print_tensor(res)

    def test_divide(self):
        import tensorflow as tf
        a = tf.constant([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ], dtype=tf.float32)
        b = tf.reduce_sum(a, axis=1)
        z = tf.reshape(b, (-1, 1))
        K.print_tensor(z)
        c = a / tf.reshape(b, (-1, 1))
        K.print_tensor(c)

    def test_model_lambdarank(self):
        model = Sequential()
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        loss = PairwiseLoss(2, 4)
        model.compile(optimizer='adam', loss=loss)

        X = K.constant([
            [0, 0], [0, 1], [1, 0], [1, 1]
        ])

        Y = K.constant([[1, 0.9], [1, 0], [0, 1], [0.1, 0.0]])
        model.fit(X, Y, epochs=10000)
        result = model.predict(X)
        K.print_tensor(result)

if __name__ == "__main__":
    unittest.main()
