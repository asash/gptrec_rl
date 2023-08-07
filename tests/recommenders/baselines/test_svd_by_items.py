import unittest

class TestSvdRecommender(unittest.TestCase):
    def test_svd_recommender_by_items(self):
        from aprec.recommenders.svd import SvdRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens1m import get_movielens1m_actions
        from aprec.datasets.movielens1m import get_movies_catalog

        catalog = get_movies_catalog()
        all_actions = get_movielens1m_actions()
        svd_recommender = SvdRecommender(256, random_seed=31337, ignore_bias=False)
        recommender = FilterSeenRecommender(svd_recommender)
        for item in all_actions:
            svd_recommender.add_action(item)
        recommender.rebuild_model()
        result = recommender.recommender.recommend_by_items('1249', 10)
        for item, score in result:
            print(catalog.get_item(item), "\t", score)
        
 






if __name__ == "__main__":
    unittest.main()


