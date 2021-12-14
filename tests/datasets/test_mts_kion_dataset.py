import os
import unittest
import json
from aprec.datasets.mts_kion import get_mts_kion_dataset, get_submission_users


class TestMtsKionDataset(unittest.TestCase):
    def test_get_mts_kion(self):
        local_path = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(local_path, "mts_kion_reference_actions.json")) as reference_file:
            reference_data = json.load(reference_file)
        data = [json.loads(action.to_json()) for action in get_mts_kion_dataset(max_actions=10)]
        self.assertEqual(reference_data, data)


    def test_get_users(self):
        submission_users = set(get_submission_users())
        dataset = get_mts_kion_dataset()
        user_ids = set([action.user_id for action in dataset])
        print(f"num_submission_users:{len(submission_users)}")
        print(f"num_submission_cold_start_users:{len(submission_users - user_ids)}")




if __name__ == '__main__':
    unittest.main()
