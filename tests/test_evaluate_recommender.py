from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.recommenders.top_recommender import TopRecommender
from aprec.utils.generator_limit import generator_limit
from aprec.evaluation.split_actions import split_actions
from aprec.evaluation.n_actions_for_user import n_actions_for_user
from aprec.evaluation.evaluate_recommender import evaluate_recommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall

import unittest
class TestEvaluateRecommender(unittest.TestCase):
    def test_evaluate(self):
        recommender = TopRecommender()
        actions = generator_limit(get_movielens_actions(), 10000)
        train, test = split_actions(actions, (70, 30))
        test = n_actions_for_user(test, 1)
        for action in train:
            recommender.add_action(action)
        recommender.rebuild_model()
        metrics = [Precision(1), Recall(1), Precision(5), Recall(5), Precision(10), Recall(10)]
        result = evaluate_recommender(recommender, test, metrics)
        reference_result = {'precision@1': 0.0, 'recall@1': 0.0,
                            'precision@5': 0.00425531914893617, 'recall@5': 0.02127659574468085,
                            'precision@10': 0.002127659574468085, 'recall@10': 0.02127659574468085}
        self.assertEqual(reference_result, result)

        
if __name__ == "__main__":
    unittest.main()


