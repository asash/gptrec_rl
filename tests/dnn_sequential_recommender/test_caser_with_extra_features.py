import unittest


class TestCaserWithExtraFeatures(unittest.TestCase):
    def test_caser_extra_features(self):
        from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
        from aprec.recommenders.dnn_sequential_recommender.featurizers.hashing_featurizer import HashingFeaturizer
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        import aprec.datasets.mts_kion as kion



        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        user_featurizer = HashingFeaturizer()
        item_featurizer = HashingFeaturizer()
        model = Caser(user_extra_features=True, requires_user_id=False)
        recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=10,
                                               users_featurizer=user_featurizer,
                                               items_featurizer=item_featurizer
                                               )
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        users = kion.get_users()
        items = kion.get_items()
        for action in kion.get_mts_kion_dataset(1000):
            recommender.add_action(action)
        for user in users:
            recommender.add_user(user)

        for item in items:
            recommender.add_item(item)

        recommender.rebuild_model()
        USER_ID = '120'
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()
