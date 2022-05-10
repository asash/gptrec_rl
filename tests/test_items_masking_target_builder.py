import unittest
import numpy as np

from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder 

class TestItemsMaskingTargetBuilder(unittest.TestCase):
    def test_target_builder(self):
        targets_builder = ItemsMaskingTargetsBuilder(relative_positions_encoding=False)
        targets_builder.set_sequence_len(5)
        targets_builder.set_n_items(10)
        targets_builder.build([(4, [(1, (1, 3)), (3, (3, 5))]), (3, [(1, (1, 6))])])
        expected_targets = np.array([[-100, 3, -100, 5, -100], [-100, 6, -100, -100, -100]])
        expected_positions = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        extra_inputs, target = targets_builder.get_targets(0, 2)
        self.assertTrue(np.all(expected_targets == target))
        self.assertTrue(np.all(extra_inputs[0] == target))
        self.assertTrue(np.all(extra_inputs[1] == expected_positions))

if __name__ == "__main__":
    unittest.main()

