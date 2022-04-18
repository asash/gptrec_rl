import unittest
from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
from aprec.recommenders.bert4recrepro.recbole_bert4rec import RecboleBERT4RecRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.utils.generator_limit import generator_limit


class TestRecboleBert4rec(unittest.TestCase):
    def test_recbole_bert4rec(self):
        USER_ID = '120'
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        recommender = RecboleBERT4RecRecommender(epochs=5, max_sequence_len=10)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

if __name__ == "__main__":
    unittest.main()