from unittest import TestCase
from aprec.datasets.movielens1m import reproduce_ber4rec_preprocessing, get_genre_dict
import unittest

class TestMovielensBert4recCopy(TestCase):
    def test_movielens_bert4rec_copy(self):
        # movielens_bert4rec_copy()
        actions = reproduce_ber4rec_preprocessing()
        genres = get_genre_dict() 
        # toy story
        print(genres['1'])



        
if __name__ == '__main__':
    unittest.main()
    