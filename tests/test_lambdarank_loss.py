from aprec.recommenders.losses.lambdarank import  PairwiseLoss
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import unittest

class TestLambdarankLoss(unittest.TestCase):
    def test_lambdas_sample(self, y, s, lambdas):
        y_true = K.constant(y)
        y_pred = K.constant(s)
        expected_lambdas = lambdas
        loss = PairwiseLoss(len(y_true[0]), len(y_true), 1)
        lambdas = loss.get_lambdas(y_true, y_pred)
        K.print_tensor(lambdas)
        eps = 1e-4
        res = tf.reduce_sum(tf.abs(lambdas - expected_lambdas))
        assert (res < eps)

    def test_get_lambdas(self):
        self.test_lambdas_sample([[0, 0, 1, 0], [0, 0, 1, 0]],
                                 [[0.1, 0.3, 1, 0], [0.5, 0, 0.5, 0]],
                                 [[0.160353, 0.174145, -0.487562, 0.153063], [2.59696, 0.0136405, -2.63147, 0.0208627]])
        self.test_lambdas_sample([[0, 0, 1, 0]], [[0.1, 0.3, 1, 0]], [[0.160353, 0.174145, -0.487562, 0.153063]])
        self.test_lambdas_sample([[0, 0, 1, 0]],[[0.5, 0, 0.5, 0]], [[2.59696, 0.0136405, -2.63147, 0.0208627]])


    def test_dcg(self):
        loss = PairwiseLoss(4, 2)
        res = loss.inverse_idcg(K.constant([0, 0, 0, 1]))
        assert res == 1

    def test_model_lambdarank(self):
        model = Sequential()
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        loss = PairwiseLoss(2, 2, sigma=1)
        model.compile(optimizer='adam', loss=loss)
        X = K.constant([[0, 0], [1, 0]])
        Y = K.constant([[1, 0],  [0, 1]])
        model.fit(X, Y, epochs=1000,verbose=False)
        result = model.predict(X)
        assert(result[0,0] > result [0, 1])
        assert(result[1,0] < result [1, 1])

if __name__ == "__main__":
    unittest.main()
