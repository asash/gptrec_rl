import os
import unittest

from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.booking_recommender.booking_recommender import BookingRecommender


class TestBookingRecommender(unittest.TestCase):
    def test_booking_recommender(self):
        current_dir = os.path.dirname(__file__)
        booking_train_file = os.path.join(current_dir, "booking_train_dataset.csv")
        booking_test_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset, submit = get_booking_dataset(booking_train_file, booking_test_file)
        recommender = BookingRecommender(train_epochs=10, n_val_users=73, batch_size=2, max_history_len=15,
                                         loss='lambdarank', output_layer_activation='linear')
        for action in dataset:
            recommender.add_action(action)
        recommender.rebuild_model()
        result = recommender.get_next_items(dataset[0].user_id, 10, dataset[0])
        print(result)

if __name__ == "__main__":
    unittest.main()