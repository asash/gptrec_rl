import unittest
import numpy as np


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
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder, RandomNegativesSampler 
        n_items = 10
        negatves_sampler = RandomNegativesSampler(3) 
        targets_builder = ItemsMaskingTargetsBuilder(relative_positions_encoding=False, negatives_sampler=negatves_sampler)
        targets_builder.set_sequence_len(5)
        targets_builder.set_n_items(n_items)
        targets_builder.build([(4, [(1, (1, 3)), (3, (3, 5))]), (3, [(1, (1, 6))])])
        expected_targets = np.array([[-100, 3, -100, 5, -100], [-100, 6, -100, -100, -100]])
        expected_positions = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        expected_negatives = np.array([[[-100, -100, -100], [   8,    2,    4], [-100, -100, -100], [   9,    0,   3], [-100, -100, -100]],
                                       [[-100, -100, -100],  [   2,    5,    1],  [-100, -100, -100], [-100, -100, -100],  [-100, -100, -100]]])
        extra_inputs, target = targets_builder.get_targets(0, 2)
        self.assertEquals(len(extra_inputs), 3) # labels, positions, sampled negatives
        sampled_negatives = extra_inputs[2]
        self.assertTrue(np.all(expected_negatives == sampled_negatives))
        self.assertEqual(len(extra_inputs), 3)
        self.assertTrue(np.all(expected_targets == target))
        self.assertTrue(np.all(extra_inputs[0] == target))
        self.assertTrue(np.all(extra_inputs[1] == expected_positions))


if __name__ == "__main__":
    unittest.main()

