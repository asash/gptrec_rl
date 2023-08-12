from collections import Counter
from unittest import TestCase
from aprec.datasets.movielens1m import reproduce_ber4rec_preprocessing, get_movies_catalog, get_genre_dict
import unittest

class TestMovielensBert4recCopy(TestCase):
    def test_movielens_bert4rec_copy(self):
        # movielens_bert4rec_copy()
        dataset = reproduce_ber4rec_preprocessing()
        genres = get_genre_dict() 
        # toy story
        print(genres['1'])
        actions_counter = Counter()
        for action in dataset:
            actions_counter[action.item_id] += 1
        
        catalog = get_movies_catalog()
        genres_dict = get_genre_dict()
        for item_id, count in actions_counter.most_common(10):
            print(f"{item_id} {catalog.get_item(item_id).title} ) {genres_dict[item_id]} {count}"),



        
if __name__ == '__main__':
    unittest.main()
    