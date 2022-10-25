import sys
import unittest

class TestSVDSampler(unittest.TestCase):
    def test_svd_sampler(self):
        from aprec.datasets.movielens20m import get_movies_catalog
        from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers import SVDSimilaritySampler 
        from aprec.tests.ml_sequences import ml_sequences
        sequences, item_ids = ml_sequences(10000)
        items_sampler = SVDSimilaritySampler(5, ann_sampling_factor=2) 
        items_sampler.set_train_sequences(sequences)
        items_sampler.set_n_items(item_ids.size())
        catalog = get_movies_catalog()
        
        positive = item_ids.get_id('1') # Toy Story
        negatives = [item_ids.reverse_id(id) for id in items_sampler.sample_negatives(positive)]
        print(negatives)
        print("sampled for Toy Story")
        for item in negatives:
            sys.stdout.write(catalog.get_item(item).title + "\n")
        print("\n")


        #self.assertEqual(negatives, ['32', '3114', '786', '1073', '736']) #Twelve Monkeys, Toy Story 2,Eraser,title=Willy Wonka & the Chocolate Factory,Twister 
        print("sampled for star wars:")
        for i in range(10):
            positive = item_ids.get_id('260') # Star Wars EP VI 
            negatives = [item_ids.reverse_id(id) for id in items_sampler.sample_negatives(positive)]
            print(negatives)
            for item in negatives:
                sys.stdout.write(catalog.get_item(item).title + "\n")
            print("\n")

if __name__ == "__main__":
    unittest.main()


