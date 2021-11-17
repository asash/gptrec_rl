from aprec.recommenders.matrix_factorization import MatrixFactorizationRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
import unittest

USER_ID = '120'

class TestMatrixFactorizationRecommender(unittest.TestCase):
    def test_matrix_factorization_recommender_recommender(self):
        for loss in ['binary_crossentropy', 'bpr', 'lambdarank', 'xendcg', 'climf']:
            print(f"testing matrix factorization model with {loss} loss")
            matrix_factorization_recommender = MatrixFactorizationRecommender(32, 5, loss, batch_size=10)
            recommender = FilterSeenRecommender(matrix_factorization_recommender)
            for action in generator_limit(get_movielens20m_actions(), 10000):
                recommender.add_action(action)
            recommender.rebuild_model()
            recs = recommender.get_next_items(USER_ID, 10)
            print(recs)



if __name__ == "__main__":
    unittest.main()
