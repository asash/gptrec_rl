import os
import unittest
from copy import deepcopy

from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.booking_recommender.booking_recommender_ltr import BookingRecommenderLTR


def LTR(model_type, attention, lgbm_objecitve='lambdarank', lgbm_boosting_type='gbdt'):
    return BookingRecommenderLTR(batch_size=2, n_val_users=10,
                                 candidates_cnt=100, val_epoch_size=10, epoch_size=100,
                                 num_training_samples=50, model_type=model_type, attention=attention,
                                 lgbm_objective=lgbm_objecitve, lgbm_boosting_type=lgbm_boosting_type,
                                 num_epochs=2)

class TestBookingLtrRecommender(unittest.TestCase):
    def test_booking_ltr_recommender(self):
        current_dir = os.path.dirname(__file__)
        booking_train_file = os.path.join(current_dir, "booking_train_dataset.csv")
        booking_test_file = os.path.join(current_dir, "booking_test_dataset.csv")
        dataset, submit = get_booking_dataset(booking_train_file, booking_test_file)
        recommenders = {}
        for boosting_type in ('rf', 'gbdt', 'dart', 'goss'):
            for objective in ('regression', 'lambdarank', 'rank_xendcg', 'regression_l1', 'huber',
                              'fair', 'poisson', 'quantile', 'mape', 'tweedie'):
                recommenders[f"Lightgbm-{boosting_type}-{objective}"] = lambda boosting_type=boosting_type, objective=objective: LTR('lightgbm', False,
                                                                                    lgbm_boosting_type=boosting_type,
                                                                                    lgbm_objecitve=objective)
        for recommender_name in recommenders:
            print(f"testing {recommender_name}")
            recommender = recommenders[recommender_name]()
            for action in dataset:
                recommender.add_action(action)
            recommender.rebuild_model()
            result = recommender.get_next_items(dataset[0].user_id, 10, dataset[0])
            print(result)

if __name__ == "__main__":
    unittest.main()
