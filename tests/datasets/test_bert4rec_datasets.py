from collections import Counter
import unittest

class TestBert4recDatasets(unittest.TestCase):
    def test_bert4rec_dataset(self):
        import json
        from aprec.datasets.dataset_stats import dataset_stats
        from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset, ml1m_mapping_to_original, get_movielens1m_catalog, get_movielens1m_genres

        # for dataset_name in ["beauty", "steam", "ml-1m"]:
        #     print(f"analyzing dataset {dataset_name}")
        #     dataset = get_bert4rec_dataset(dataset_name)
        #     stats = dataset_stats(dataset, metrics=['num_users', 'num_items', 'num_interactions'])
        #     print(json.dumps(stats, indent=4))

        dataset = get_bert4rec_dataset("ml-1m")
        actions_counter = Counter()
        for action in dataset:
            actions_counter[action.item_id] += 1
        
        catalog = get_movielens1m_catalog()
        genres_dict = get_movielens1m_genres()
        mapping = ml1m_mapping_to_original()
        for item_id, count in actions_counter.most_common(10):
            print(f"{item_id} {mapping[item_id]} {catalog.get_item(item_id).title} ) {genres_dict[item_id]}", count)
        
        

if __name__ == "__main__":
    unittest.main()