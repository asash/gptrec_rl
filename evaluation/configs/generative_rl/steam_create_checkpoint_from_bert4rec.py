from aprec.datasets.datasets_register import DatasetsRegister
from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.pcount import PCOUNT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.fmc_plus import SmartMC
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward
from aprec.datasets.steam import get_genres_steam_deduped_1000items_warm_users



USERS_FRACTIONS = [1.0]
genre_func = get_genres_steam_deduped_1000items_warm_users
DATASET = "steam_deduped_1000items_warm_users_noties"

METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()), PCOUNT(10, DatasetsRegister()[DATASET]()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200
#checkpoints will be created in any case
SAVE_MODELS=False
RECOMMENDATIONS_LIMIT=100


def generative_tuning_recommender_from_bert(ild_lambda, checkpoint, max_pretrain_epochs=10000):       
        from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.dummy_builder import DummyTargetBuilder
        from aprec.recommenders.rl_generative.pre_train_target_splitter import PreTrainTargetSplitter
        from aprec.recommenders.rl_generative.teacher import TeacherRecommender
        pretrain_recommender=TeacherRecommender(checkpoint)
        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=1500, attention_heads=4)
        pre_training_recommender = lambda: pretrain_recommender

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=max_pretrain_epochs,
                                               early_stopping=True,
                                               early_stop_epochs=200,
                                               batch_size=128,
                                               training_time_limit=200000,  
                                               sequence_splitter=PreTrainTargetSplitter, 
                                               max_batches_per_epoch=100,
                                               targets_builder=DummyTargetBuilder,
                                               use_keras_training=True,
                                               sequence_length=SEQUENCE_LENGTH,
                                               validate_on_loss=True, 
                                               train_on_val_users=False 
                                               )
        recommender = GenerativeTuningRecommender(recommender_config, pre_training_recommender,
                                                  validate_every_steps=500, max_tuning_steps=0, 
                                                  tuning_batch_size=16, 
                                                  clip_eps=0.1,
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILDReward(genre_func())], [1, ild_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILDReward(genre_func()))],
                                                  gae_gamma=0.1, 
                                                  gae_lambda=0.1, 
                                                  internal_pretrain=True, #note that training parameters from recommender config won't be used
                                                  internal_pretrain_max_batches=1000 * 100, #100 batches per epoch, 1000 epochs
                                                  )
        return recommender
        
        
recommenders = {
    
} 

recommenders[f"generative_tuning_recommender_pretrain_bert4rec"] = lambda: generative_tuning_recommender_from_bert(0.0, checkpoint="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/steam_baselines/bert4rec.dill.gz")
    

def get_recommenders(filter_seen: bool):
    result = {}
    all_recommenders = list(recommenders.keys())
    for recommender_name in all_recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result


N_VAL_USERS=512
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)

