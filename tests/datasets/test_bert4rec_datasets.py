import json
from collections import defaultdict

import numpy as np

from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
import unittest

class TestBert4recDatasets(unittest.TestCase):
    def test_bert4rec_dataset(self):
        for dataset_name in ["beauty", "steam", "ml-1m"]:
            print(f"analyzing dataset {dataset_name}")
            dataset = get_bert4rec_dataset(dataset_name)
            print(json.dumps(self.get_stats(dataset), indent=4))

    def get_stats(self, dataset):
        users = defaultdict(list)
        item_ids = set()
        for action in dataset:
            users[action.user_id].append(action)
            item_ids.add(action.item_id)
        session_lens = [len(users[user_id]) for user_id in users]
        result = {
            "num_users": len(users),
            "num_items": len(item_ids),
            "average_session_len": float(np.mean(session_lens)),
            "median_session_len": int(np.median(session_lens)),
            "max_session_len": int(np.max(session_lens)),
            "min_session_len": int(np.min(session_lens)),
            "p80_session_len": float(np.percentile(session_lens, 80))
        }
        return result




if __name__ == "__main__":
    unittest.main()