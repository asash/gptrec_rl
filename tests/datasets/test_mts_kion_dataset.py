import os
import unittest
import json
from aprec.datasets.mts_kion import get_mts_kion_dataset, get_submission_user_ids, get_users


class TestMtsKionDataset(unittest.TestCase):
    def test_get_mts_kion(self):
        local_path = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(local_path, "mts_kion_reference_actions.json")) as reference_file:
            reference_data = json.load(reference_file)
        data = [json.loads(action.to_json()) for action in get_mts_kion_dataset(max_actions=10)]
        self.assertEqual(reference_data, data)


    def test_get_submission_user_ids(self):
        submission_users = get_submission_user_ids()
        self.assertEqual(submission_users[:10],
                         ['3', '11', '29', '30', '33', '39', '46', '47', '51', '61'])

    def test_get_users(selfs):
        users = get_users()
        pass



if __name__ == '__main__':
    unittest.main()
