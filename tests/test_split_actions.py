import unittest
from aprec.tests.generate_actions import generate_actions
from aprec.evaluation.split_actions import split_actions

class TestSplitActions(unittest.TestCase):

    def test_split_actions(self):
        actions =  generate_actions(100)
        splitted = split_actions(actions, (7, 1, 2))
        self.assertEqual(len(splitted), 3)
        self.assertEqual(len(splitted[0]), 70)
        self.assertEqual(len(splitted[1]), 10)
        self.assertEqual(len(splitted[2]), 20)
        assert(times_func(splitted[0], max) <= times_func(splitted[1], min))
        assert(times_func(splitted[1], max) <= times_func(splitted[2], min))
        self.assertEqual(set(actions), set(splitted[0] + splitted[1] + splitted[2]))

def times_func(actions, func):
    return func([action.timestamp for action in actions])
        
if __name__ == "__main__":
    unittest.main()
