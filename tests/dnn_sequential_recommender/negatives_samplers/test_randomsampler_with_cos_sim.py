from collections import Counter
import unittest
import sys

import tqdm 
class TestRandomSamplerCosSim(unittest.TestCase):
    def test_random_sampler_cos_sim(self):
        from aprec.datasets.movielens20m import get_movies_catalog
        from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers import RandomNegativesWithCosSimValues
        from aprec.tests.ml_sequences import ml_sequences
        sequences, item_ids = ml_sequences(1000000)
        items_sampler = RandomNegativesWithCosSimValues(5)
        items_sampler.set_n_items(item_ids.size())

        items_sampler.set_train_sequences(sequences)
        items_sampler.set_n_items(item_ids.size())
        catalog = get_movies_catalog()
        
        check = '8368'
        positive = item_ids.get_id(check)
        title = catalog.get_item(check).title
        best_internal, scores = items_sampler.top_sims(positive)
        print(f"{title} ({positive})")
        for i in range(10):
            item = item_ids.reverse_id(best_internal[i])
            score = scores[i]
            sys.stdout.write(str(best_internal[i]) + " " + catalog.get_item(item).title + f"score: {score} \n")
        print("\n")

        #self.assertEqual(negatives, ['32', '3114', '786', '1073', '736']) #Twelve Monkeys, Toy Story 2,Eraser,title=Willy Wonka & the Chocolate Factory,Twister 
        title_cnt = Counter()
        for i in tqdm.tqdm(range(100000)):
            sampled_negatives, scores = items_sampler.sample_negatives(positive)
            negatives = [item_ids.reverse_id(item) for item in sampled_negatives]
            for item in negatives:
                t = catalog.get_item(item).title
                title_cnt[t] += 1
        
        print(f"most frequently sampled for {title}:")
        for t, cnt in title_cnt.most_common(100):
            print(t, cnt)

if __name__ == "__main__":
    unittest.main()