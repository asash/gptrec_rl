import os
import unittest
from aprec.datasets.booking import get_booking_dataset_one_file, get_booking_dataset
from aprec.evaluation.split_actions import get_control_users, leave_one_out
from aprec.recommenders.top_recommender import TopRecommender

class TestBookingDataset(unittest.TestCase):
    def test_train_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_file = os.path.join(current_dir, "booking_train_dataset.csv")
        dataset = get_booking_dataset_one_file(booking_file, is_testset=False)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()

        recommendations = recommender.get_next_items('1010293', 5)
        assert(recommendations[0][0] == '36063')

    def test_test_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset = get_booking_dataset_one_file(booking_file, is_testset=True)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()

        recommendations = recommender.get_next_items('1010293', 5)
        self.assertEqual(recommendations[0][0], '26235')

    def test_full_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_train_file = os.path.join(current_dir, "booking_train_dataset.csv")
        booking_test_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset = get_booking_dataset(booking_train_file, booking_test_file)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()
        recommendations = recommender.get_next_items('1010293', 5)
        self.assertEqual(recommendations[0][0], '29319')
        control_trips = get_control_users(dataset)
        train, validation = leave_one_out(dataset, max_test_users=100)
        for action in validation:
            self.assertFalse(action.data['is_control'])
            self.assertFalse(action.user_id in control_trips)

