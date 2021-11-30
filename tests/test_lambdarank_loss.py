from aprec.losses.lambdarank import  LambdaRankLoss
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import unittest

class TestLambdarankLoss(unittest.TestCase):
    def lambdas_sample_test(self, y, s, lambdas, ndcg_at=30):
        y_true = K.constant(y)
        y_pred = K.constant(s)
        expected_lambdas = lambdas
        loss = LambdaRankLoss(len(y_true[0]), len(y_true), 1, ndcg_at)
        lambdas = loss.get_lambdas(y_true, y_pred)
        eps = 1e-4
        res = tf.reduce_all(tf.abs(lambdas - expected_lambdas) < eps)
        assert res

    def test_get_lambdas(self):
        self.lambdas_sample_test([[1, 1, 1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]],
                                 [[0.394383, 0.79844, 0.197551, 0.76823, 0.55397, 0.628871, 0.513401, 0.916195, 0.717297, 0.606969],
                                  [0.0738184, 0.758257, 0.502675, 0.0370191, 0.985911, 0.0580367, 0.669592, 0.666748, 0.830348, 0.252756]],
                                 [[-0.0981529, -0.586437, -0.0719084, 0.54124, -0.140295, 0.291305, 0.151254, -0.686095, 0.401098, 0.197991],
                                  [0.050076, 0.337357, 0.113701, 0.051092, -0.751471, 0.0510903, 0.507125, -0.643697, 0.41508, -0.130353]], 5)
        self.lambdas_sample_test([[0, 0, 1, 1]], [[0.1, 0.3, 1, 0]], [[0.175801, 0.190922, -0.366723, 0]], 1)
        self.lambdas_sample_test([[0, 0, 1, 1]], [[0.1, 0.3, 1, 0]], [[0.101338, 0.346829, -0.211393, -0.236774]], 2)
        self.lambdas_sample_test([[0, 0, 1, 1], [0, 0, 1, 1]],
                                [[0.1, 0.3, 1, 0], [0.1, 0.3, 1, 0]],
                                [[0.101338, 0.346829, -0.211393, -0.236774], [0.101338, 0.346829, -0.211393, -0.236774]]
                                 , 2)


        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.5, 0, 0.5, 0]], [[2.59696, 0.0136405, -2.63147, 0.0208627]], 2)
        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.1, 0.3, 1, 0]], [[0.160353, 0.174145, -0.487562, 0.153063]], 1)
        self.lambdas_sample_test([[0, 0, 1, 0], [0, 0, 1, 0]],
                                 [[0.1, 0.3, 1, 0], [0.5, 0, 0.5, 0]],
                                 [[0.160353, 0.174145, -0.487562, 0.153063], [2.59696, 0.0136405, -2.63147, 0.0208627]])
        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.1, 0.3, 1, 0]], [[0.160353, 0.174145, -0.487562, 0.153063]])
        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.5, 0, 0.5, 0]], [[2.59696, 0.0136405, -2.63147, 0.0208627]])


    def test_dcg(self):
        loss = LambdaRankLoss(4, 1, ndcg_at=1)
        res = loss.get_inverse_idcg(K.constant([[0, 0, 1, 1]]))
        assert res == 1

        loss = LambdaRankLoss(4, 1)
        res = loss.get_inverse_idcg(K.constant([[0, 0, 0, 1]]))
        assert res == 1



    def test_model_lambdarank(self):
        model = Sequential()
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        loss = LambdaRankLoss(2, 2, sigma=1)
        model.compile(optimizer='adam', loss=loss)
        X = K.constant([[0, 0], [1, 0]])
        Y = K.constant([[1, 0],  [0, 1]])
        model.fit(X, Y, epochs=1000,verbose=False)
        result = model.predict(X)
        assert(result[0,0] > result [0, 1])
        assert(result[1,0] < result [1, 1])

if __name__ == "__main__":
    unittest.main()
