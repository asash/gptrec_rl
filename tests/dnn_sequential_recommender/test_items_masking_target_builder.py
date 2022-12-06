import unittest
import numpy as np
import sys


class TestItemsMaskingTargetBuilder(unittest.TestCase):
    def test_target_builder(self):
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
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
        from aprec.recommenders.sequential.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder
        from aprec.recommenders.sequential.target_builders.negative_samplers import RandomNegativesSampler 
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



if __name__ == "__main__":
    unittest.main()

