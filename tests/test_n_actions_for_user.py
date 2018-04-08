import unittest
from aprec.tests.generate_actions import generate_actions
from aprec.evaluation.n_actions_for_user import n_actions_for_user

REFERENCE_1_ACTION =\
"""user_id=0, item_id=0, timestamp=2, data={}
user_id=1, item_id=3, timestamp=0, data={}
user_id=2, item_id=0, timestamp=0, data={}
user_id=3, item_id=2, timestamp=3, data={}"""


REFERENCE_2_ACTION =\
"""user_id=0, item_id=0, timestamp=2, data={}
user_id=0, item_id=2, timestamp=4, data={}
user_id=1, item_id=0, timestamp=0, data={}
user_id=1, item_id=3, timestamp=0, data={}
user_id=2, item_id=0, timestamp=0, data={}
user_id=2, item_id=2, timestamp=2, data={}
user_id=3, item_id=0, timestamp=4, data={}
user_id=3, item_id=2, timestamp=3, data={}"""

def sorted_actions_str(actions):
    return "\n".join(sorted([str(action) for action in actions]))

class TestNActionsForUser(unittest.TestCase):
    def test_n_actions_for_user(self):
        actions = generate_actions(10)
        actions_1 = n_actions_for_user(actions, 1)
        actions_2 = n_actions_for_user(actions, 2)
        self.assertEqual(sorted_actions_str(actions_1), REFERENCE_1_ACTION)
        self.assertEqual(sorted_actions_str(actions_2), REFERENCE_2_ACTION)
        

if __name__ == "__main__":
    unittest.main()
