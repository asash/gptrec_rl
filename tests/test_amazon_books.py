import unittest
from aprec.datasets.amazon_books import download_amazon_books_dataset, extract_amazon_books_dataset, \
    get_amazon_books_dataset


class TestAmazonBooks(unittest.TestCase):
    def test_amazon_books(self):
        return
        download_amazon_books_dataset()
        extract_amazon_books_dataset()
        actions = get_amazon_books_dataset(min_users_per_item=20, min_actions_per_user=5)
        items_set = set()
        users_set = set()
        for action in actions:
            items_set.add(action.item_id)
            users_set.add(action.user_id)
        print("n_users:", len(items_set), "n_items:", len(items_set),"n_actions:", len(actions))