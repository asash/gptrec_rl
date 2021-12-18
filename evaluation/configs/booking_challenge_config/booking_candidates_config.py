from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.sps import SPS
from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.booking_recommender.candidates_recommender import BookingCandidatesRecommender

DATASET, SUBMIT_ACTIONS = get_booking_dataset()



RECOMMENDERS = {
    "booking_candidates": BookingCandidatesRecommender
}

SPLIT_STRATEGY = "LEAVE_ONE_OUT"
USERS_FRACTIONS = [1.0]
METRICS = [Precision(4), SPS(4), NDCG(4), NDCG(40), SPS(50), SPS(100), SPS(150), SPS(200), SPS(300), SPS(400), SPS(500), SPS(600), SPS(700), SPS(800), SPS(900), SPS(1000)]
