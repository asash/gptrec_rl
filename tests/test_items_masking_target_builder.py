import unittest
import numpy as np
import sys


class TestItemsMaskingTargetBuilder(unittest.TestCase):
    def test_target_builder(self):
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        targets_builder = ItemsMaskingTargetsBuilder(relative_positions_encoding=False)
        targets_builder.set_sequence_len(5)
        targets_builder.set_n_items(10)
        targets_builder.build([(4, [(1, (1, 3)), (3, (3, 5))]), (3, [(1, (1, 6))])])
        expected_targets = np.array([[-100, 3, -100, 5, -100], [-100, 6, -100, -100, -100]])
        expected_positions = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        extra_inputs, target = targets_builder.get_targets(0, 2)
        self.assertEqual(len(extra_inputs), 2)
        self.assertEquals(len(extra_inputs), 2) # labels, positions
        self.assertTrue(np.all(expected_targets == target))
        self.assertTrue(np.all(extra_inputs[0] == target))
        self.assertTrue(np.all(extra_inputs[1] == expected_positions))

    def test_target_builder_random_negatives(self):
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder, RandomNegativesSampler 
        n_items = 10
        negatves_sampler = RandomNegativesSampler(3) 
        targets_builder = ItemsMaskingWithNegativesTargetsBuilder(relative_positions_encoding=False, negatives_sampler=negatves_sampler)
        targets_builder.set_sequence_len(5)
        targets_builder.set_n_items(n_items)
        targets_builder.build([(4, [(1, (1, 3)), (3, (3, 5))]), (3, [(1, (1, 6))])])
        expected_positions = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        expected_sampled_items = np.array([[[12, 12, 12, 12], [3,    8,    2,    4], [12, 12, 12, 12], [5,   9,    0,   3], [12, 12, 12, 12]],
                                       [[12, 12, 12, 12],  [ 6,  2,    5,    1],  [12, 12, 12, 12], [12, 12, 12, 12],  [12, 12, 12, 12]]])
        expected_target = np.array([[[-100, -100, -100, -100], [   1,    0,    0,    0], [-100, -100, -100, -100], [   1,    0,    0,    0], [-100, -100, -100, -100]],
                                    [[-100, -100, -100, -100], [   1,    0,    0,    0],  [-100, -100, -100, -100],[-100, -100, -100, -100], [-100, -100, -100, -100]]])
        extra_inputs, target = targets_builder.get_targets(0, 2)
        sampled_items = extra_inputs[0]

        self.assertEqual(len(extra_inputs), 2)
        self.assertTrue(np.all(target == expected_target))
        self.assertTrue(np.all(expected_sampled_items == sampled_items))
        self.assertTrue(np.all(extra_inputs[1] == expected_positions))

    @staticmethod
    def ml_sequences(n_actions):
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.item_id import ItemId
        from collections import defaultdict
        sequences_dict = defaultdict(list)
        actions = [action for action in generator_limit(get_movielens20m_actions(), n_actions)]
        actions.sort(key = lambda action: action.timestamp)
        item_ids = ItemId()
        for action in actions:
            sequences_dict[action.user_id].append((action.timestamp, item_ids.get_id(action.item_id)))
        sequences = list(sequences_dict.values())
        return sequences, item_ids


    def test_hard_items(self):
        from aprec.datasets.movielens20m import get_movies_catalog
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import SVDSimilaritySampler 
        ml_sequences, item_ids = self.ml_sequences(10000)
        items_sampler = SVDSimilaritySampler(5, ann_sampling_factor=2) 
        items_sampler.set_train_sequences(ml_sequences)
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





print("\n")



if __name__ == "__main__":
    unittest.main()

