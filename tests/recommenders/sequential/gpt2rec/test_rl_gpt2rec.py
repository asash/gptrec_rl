import os
import unittest
from aprec.recommenders.sequential.generative_tuning_recommender import GenerativeTuningRecommender
from aprec.datasets.datasets_register import DatasetsRegister

from aprec.recommenders.top_recommender import TopRecommender


class TestRLGptRec(unittest.TestCase):
    def setUp(self):
        pass
#        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    def test_RLGPTRec(self):
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.sequential.targetsplitters.id_splitter import IdSplitter
        from aprec.recommenders.sequential.target_builders.dummy_builder import DummyTargetBuilder
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.utils.generator_limit import generator_limit

        USER_ID = '4493'
        val_users = ['52345', '39828', '42989', '84704', '26479', '9911', '67788', '120329', '42797', '28082']
        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=64, tokenizer='id', tokens_per_item=1, values_per_dim=55, attention_heads=4)
        pre_training_recommender = lambda: TopRecommender()

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=10,
                                               training_time_limit=20,  
                                               sequence_splitter=IdSplitter, 
                                               max_batches_per_epoch=100,
                                               targets_builder=DummyTargetBuilder,
                                               use_keras_training=True,
                                               sequence_length=32,
                                               validate_on_loss=True
                                               )
        
        
        recommender = GenerativeTuningRecommender(recommender_config, pre_training_recommender, validate_every_steps=2, max_tuning_steps=4)

        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in DatasetsRegister()['ml-20m_50items_fraction_0.01']():
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])
        batch = [('52345', None), ('39828', None)]
        recommender.recommend_batch(batch, limit=10)

if __name__ == "__main__":
    unittest.main()