from recommenders.dnn_sequential_recommender.models.sasrec.sasrec_layer import SASRecLayer
import unittest
import random
import numpy as np

class TestSasrecLayer(unittest.TestCase):
    def test_sasrec_layer(self):
        seq_len = 50
        itemnum = 5
        layer = SASRecLayer(itemnum=5, hidden_units=30)

        random.seed(5)
        input = np.array([[random.randint(1, itemnum) for _ in range(seq_len)],
                          [random.randint(1, itemnum) for _ in range(seq_len)]
                          ])
        encoded = layer(input)
        self.assertEqual(encoded.shape, [2, itemnum])
        print(encoded)

