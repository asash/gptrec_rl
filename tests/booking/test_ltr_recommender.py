import os
import unittest

from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.booking_recommender.booking_recommender_ltr import BookingRecommenderLTR


class TestBookingLtrRecommender(unittest.TestCase):
    def test_booking_ltr_recommender(self):
        current_dir = os.path.dirname(__file__)
        booking_train_file = os.path.join(current_dir, "booking_train_dataset.csv")
        booking_test_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset, submit = get_booking_dataset(booking_train_file, booking_test_file)
        recommender = BookingRecommenderLTR(n_val_users=73, batch_size=2, val_epoch_size=50,
                                            epoch_size=10, num_training_samples=5000, model_type='neural')
        for action in dataset:
            recommender.add_action(action)
        recommender.rebuild_model()
        result = recommender.get_next_items(dataset[0].user_id, 10, dataset[0])
        print(result)

if __name__ == "__main__":
    unittest.main()
