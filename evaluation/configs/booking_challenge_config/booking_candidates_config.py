from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.booking_recommender.candidates_recommender import BookingCandidatesRecommender
from aprec.evaluation.split_actions import LeaveOneOut

DATASET, SUBMIT_ACTIONS = get_booking_dataset()



RECOMMENDERS = {
    "booking_candidates": BookingCandidatesRecommender
}

SPLIT_STRATEGY = "LEAVE_ONE_OUT"
USERS_FRACTIONS = [1.0]
METRICS = [Precision(4), HIT(4), NDCG(4), NDCG(40), HIT(50), HIT(100), HIT(150), HIT(200), HIT(300), HIT(400), HIT(500), HIT(600), HIT(700), HIT(800), HIT(900), HIT(1000)]
