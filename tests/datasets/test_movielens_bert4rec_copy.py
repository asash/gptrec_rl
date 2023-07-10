from unittest import TestCase
from aprec.datasets.movielens1m import get_movielens1m_actions, reproduce_ber4rec_preprocessing
import unittest

class TestMovielensBert4recCopy(TestCase):
    def test_movielens_bert4rec_copy(self):
        # movielens_bert4rec_copy()
        actions = reproduce_ber4rec_preprocessing()
        return actions


        
if __name__ == '__main__':
    unittest.main()
    