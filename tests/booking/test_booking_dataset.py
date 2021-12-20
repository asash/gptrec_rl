import os
import unittest
from aprec.datasets.booking import get_booking_dataset_one_file, get_booking_dataset_internal
from aprec.evaluation.split_actions import get_control_users, LeaveOneOut
from aprec.recommenders.top_recommender import TopRecommender


class TestBookingDataset(unittest.TestCase):
    def test_train_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_file = os.path.join(current_dir, "booking_train_dataset.csv")
        dataset, submit = get_booking_dataset_one_file(booking_file, is_testset=False)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()

        recommendations = recommender.recommend('1010293', 5)
        self.assertEqual(recommendations[0][0], '23921')

    def test_test_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset, submit = get_booking_dataset_one_file(booking_file, is_testset=True)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()

        recommendations = recommender.recommend('1010293', 5)
        self.assertEqual(recommendations[0][0], '26235')

    def test_full_dataset(self):
        current_dir = os.path.dirname(__file__)
        booking_train_file = os.path.join(current_dir, "booking_train_dataset.csv")
        booking_test_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset, submit = get_booking_dataset_internal(booking_train_file, booking_test_file)
        recommender = TopRecommender()
        for action in dataset:
            recommender.add_action(action)
            recommender.rebuild_model()
        recommendations = recommender.recommend('1010293', 5)
        self.assertEqual(recommendations[0][0], '29319')
        control_trips = get_control_users(dataset)
        splitter = LeaveOneOut(max_test_users=100)
        train, validation = splitter(dataset)
        for action in validation:
            self.assertFalse(action.data['is_control'])
            self.assertFalse(action.user_id in control_trips)

if __name__ == "__main__":
    unittest.main()
