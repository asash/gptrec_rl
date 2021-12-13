import os
import unittest

from aprec.datasets.booking import get_booking_dataset_internal
from aprec.recommenders.booking_recommender.candidates_recommender import BookingCandidatesRecommender


class TestCandidatesRecommender(unittest.TestCase):
    def test_candadates_recommender(self):
        current_dir = os.path.dirname(__file__)
        booking_train_file = os.path.join(current_dir, "booking_train_dataset.csv")
        booking_test_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset, submit = get_booking_dataset_internal(booking_train_file, booking_test_file)
        recommender = BookingCandidatesRecommender()
        for action in dataset:
            recommender.add_action(action)
        recommender.rebuild_model()
        result = recommender.get_next_items(dataset[0].user_id, 10)
        trip = list(filter(lambda item: item.user_id == dataset[0].user_id, dataset))
        candidates = recommender.get_candidates_with_features(trip, 10)
        for vector in candidates:
            print(vector)
        shape = recommender.n_features
        print(shape)

if __name__ == "__main__":
    unittest.main()
