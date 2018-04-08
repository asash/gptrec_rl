from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.utils import generator_limit
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall


DATASET = generator_limit(get_movielens_actions(min_rating=4.0), 100000)
RECOMMENDERS = {
    "top_recommender": lambda: TopRecommender()    
}

FRACTIONS_TO_SPLIT = (0.2, 0.4, 0.6, 0.8)
TEST_ACTION_PER_USER = 1
METRICS = [Precision(1), Recall(1), Precision(5), Recall(5)]
