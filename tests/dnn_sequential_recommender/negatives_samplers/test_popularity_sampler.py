import sys
from collections import Counter
import time
import unittest

import tqdm


class TestPopularitySampler(unittest.TestCase):
    def test_popularity_sampler(self):
        from aprec.datasets.movielens20m import get_movies_catalog
        from aprec.recommenders.sequential.target_builders.negative_samplers.popularity_based_sampler import PopularityBasedSampler
        from aprec.tests.ml_sequences import ml_sequences
        sequences, item_ids = ml_sequences(10000)
        items_sampler = PopularityBasedSampler(5) 
        items_sampler.set_train_sequences(sequences)
        items_sampler.set_n_items(item_ids.size())
        catalog = get_movies_catalog()
        
        positive = item_ids.get_id('1') # Toy Story
        ids, scores = items_sampler.sample_negatives(positive)
        negatives = [item_ids.reverse_id(id) for id in ids]
        print(negatives)
        print("sampled for Toy Story")
        for item in negatives:
            sys.stdout.write(catalog.get_item(item).title + "\n")
        print("\n")


        #self.assertEqual(negatives, ['32', '3114', '786', '1073', '736']) #Twelve Monkeys, Toy Story 2,Eraser,title=Willy Wonka & the Chocolate Factory,Twister 
        print("sampled for star wars:")
        title_cnt = Counter()
        for i in tqdm.tqdm(range(100000)):
            sampled_negatives = items_sampler.sample_negatives(positive)
            ids, scores = items_sampler.sample_negatives(positive)
            negatives = [item_ids.reverse_id(item) for item in ids]
            for item in negatives:
                title = catalog.get_item(item).title
                title_cnt[title] += 1
        print(title_cnt.most_common(10))


if __name__ == "__main__":
    unittest.main()


