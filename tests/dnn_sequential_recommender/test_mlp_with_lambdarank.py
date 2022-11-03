import unittest


class TestMLPWithLambdarank(unittest.TestCase):
    def test_mlp_recommender_with_lambdarank(self):
        from aprec.recommenders.dnn_sequential_recommender.models.mlp_sequential import SequentialMLPModel
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        batch_size = 10
        model = SequentialMLPModel(output_layer_activation='linear')
        loss = LambdaGammaRankLoss(ndcg_at=10)
        mlp_recommender = DNNSequentialRecommender(model, loss, train_epochs=10, batch_size=batch_size)

        mlp_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        mlp_recommender.set_loss(loss)
        recommender.rebuild_model()
        USER_ID = '120'
        recs = recommender.recommend(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)

if __name__ == '__main__':
    unittest.main()