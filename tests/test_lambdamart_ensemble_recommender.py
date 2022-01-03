import unittest
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lambdamart_ensemble_recommender import LambdaMARTEnsembleRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.tests.test_deepmf import USER_ID
from aprec.utils.generator_limit import generator_limit

class TestLambdaMartEnsembleRecommender(unittest.TestCase):
    def test_lambdamart_ensemble_recommender(self):
        candidates_selection = FilterSeenRecommender(TopRecommender())
        other_recommenders = {
                                "svd_recommender": SvdRecommender(128)
                             }
        recommender = LambdaMARTEnsembleRecommender(
                            candidates_selection_recommender=candidates_selection, 
                            other_recommenders=other_recommenders,
                            n_ensemble_users=200
        ) 
        
        USER_ID = '120'

        for action in generator_limit(get_movielens20m_actions(), 100000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()



