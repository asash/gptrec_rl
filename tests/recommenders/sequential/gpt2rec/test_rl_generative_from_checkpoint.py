import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random

from aprec.recommenders.sequential.models.generative.reward_metrics.acc_reward import ACCReward
import unittest
from aprec.datasets.movielens1m import get_genre_dict, get_movies_catalog
from aprec.recommenders.fmc_plus import SmartMC
from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
from aprec.datasets.datasets_register import DatasetsRegister
from aprec.recommenders.rl_generative.pre_train_target_splitter import PreTrainTargetSplitter
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward


from aprec.recommenders.top_recommender import TopRecommender


class TestRLGptRec(unittest.TestCase):
    def setUp(self):
        pass

    def get_recommender(self, pre_training_recommender=None, checkpoint=None):
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

        val_users = ['5112', '2970', '3159', '3345', '2557', '1777', '4111', '3205', '4380', '5508']
        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=64, tokenizer='id', tokens_per_item=1, values_per_dim=55, attention_heads=4)
 

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=10,
                                               training_time_limit=20,  
                                               sequence_splitter=PreTrainTargetSplitter, 
                                               max_batches_per_epoch=100,
                                               use_keras_training=True,
                                               sequence_length=32,
                                               validate_on_loss=True,
                                               )
        
        
        recommender = GenerativeTuningRecommender(recommender_config, pre_training_recommender,
                                                  pre_trained_checkpoint_dir=checkpoint,
                                                  validate_every_steps=2,
                                                  max_tuning_steps=10,
                                                  reward_metric=WeightedSumReward([ACCReward(10), ILDReward(get_genre_dict())], [1, 0.05]),
                                                  tradeoff_monitoring_rewards = [(NDCGReward(10), ILDReward(get_genre_dict()))],
                                                  checkpoint_every_steps=3,
                                                  sampling_processessess=1, 
                                                  validate_before_tuning=False,
                                                  batch_recommendation_processess=1
                                                  )


        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        return recommender

    def get_checkpoints_dir(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_checkpoints')

    def test_from_pretrained_checkpoint(self):
        USER_ID = '22'
        catalog = get_movies_catalog()
        checkpoint_dir = self.get_checkpoints_dir()
        pretrained_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_pre_train_no_val')
        recommender = self.get_recommender(checkpoint=pretrained_checkpoint)
        actions = []
        for action in DatasetsRegister()['ml-1m_50items_fraction_0.2']():
            actions.append(action)
        random.shuffle(actions)
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])
        batch = [('6002', None), ('2591', None)]
        recs = recommender.recommend_batch(batch, limit=10)       
        print(recs)

if __name__ == "__main__":
    unittest.main()