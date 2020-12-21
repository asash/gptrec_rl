import os
import unittest
from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.top_recommender import TopRecommender

class TestBookingDataset(unittest.TestCase):
    def test_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_file = os.path.join(current_dir, "booking_train_dataset.csv")
        dataset = get_booking_dataset(booking_file)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()

        recommendations = recommender.get_next_items('1010293', 5)
        assert(recommendations[0][0] == '36063')

