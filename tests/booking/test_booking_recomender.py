import os
import unittest

from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.booking_recommender.booking_recommender import BookingRecommender


class TestBookingRecommender(unittest.TestCase):
    def test_booking_recommender(self):
        current_dir = os.path.dirname(__file__)
        booking_file = os.path.join(current_dir, "booking_train_dataset.csv")
        dataset = get_booking_dataset(booking_file)
        recommender = BookingRecommender(train_epochs=10, n_val_users=73, batch_size=2, max_history_len=15,
                                         loss='lambdarank', output_layer_activation='linear')
        for action in dataset:
            recommender.add_action(action)
        recommender.rebuild_model()
        result = recommender.get_next_items('39641', 10)
        print(result)

if __name__ == "__main__":
    unittest.main()