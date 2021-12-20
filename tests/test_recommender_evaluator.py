import tempfile

from aprec.evaluation.evaluate_recommender import RecommendersEvaluator
import unittest

from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.metrics.precision import Precision
from aprec.recommenders.top_recommender import TopRecommender


class TestRecommenderEvaluator(unittest.TestCase):
    def test_recommender_evaluator(self):
        actions = [action for action in generator_limit(get_movielens20m_actions(), 100000)]
        recommenders= {"top_recommender": TopRecommender}

        data_splitter = LeaveOneOut(max_test_users=128)
        metrics = [Precision(5)]
        out_dir = tempfile.mkdtemp()
        n_val_users=10
        recommendations_limit = 10
        evaluator = RecommendersEvaluator(actions, recommenders, metrics,
                                          out_dir, data_splitter, n_val_users,
                                          recommendations_limit, n_sampled_ranking=20)
        result = evaluator()['recommenders']['top_recommender']
        del(result["model_build_time"])
        del(result["model_inference_time"])
        self.assertEqual(result,
              {'precision@5': 0.0078125, 'sampled_metrics': {'precision@5': 0.039062500000000014},'model_metadata': {}})
        
if __name__ == "__main__":
    unittest.main()
