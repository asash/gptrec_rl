import json
from aprec.datasets.steam import get_genres_steam_deduped_1000items_warm_users

from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.pcount import PCOUNT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


from aprec.datasets.datasets_register import DatasetsRegister
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.lightfm import LightFMRecommender


USERS_FRACTIONS = [1.0]
RECOMMENDATIONS_LIMIT=500

genre_func = get_genres_steam_deduped_1000items_warm_users
DATASET = "steam_deduped_1000items_warm_users_noties"


METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()), PCOUNT(10, DatasetsRegister()[DATASET]()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)
#SAVE_MODELS = False

SEQUENCE_LENGTH=200


CHECKPOINT="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/steam_pretrained_from_bert4rec_10kstepst"

#1.0  for gae_gamma and gae_lambda allows the model to see and plan for the whole sequence
def generative_tuning_recommender(ild_lambda=0.5, checkpoint_dir=CHECKPOINT, gae_gamma=1.0, gae_lambda=1.0, max_tuning_steps=32000):       
        import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
        from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
        from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward

        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=1500, attention_heads=4)
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=0) 
        recommender = GenerativeTuningRecommender(recommender_config,
                                                  pre_trained_checkpoint_dir=checkpoint_dir,
                                                  #pre_train_recommender_factory=lambda: RandomRecommender(),
                                                  max_tuning_steps=max_tuning_steps, 
                                                  tuning_batch_size=128, 
                                                  clip_eps=1.0, #allow large changes here, most divergence is controlled by klpen
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILDReward(genre_func())], [1, ild_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILDReward(genre_func()))],
                                                  gae_gamma=gae_gamma, 
                                                  gae_lambda=gae_lambda,
                                                  validate_before_tuning=True,
                                                  sampling_processessess=8,
                                                  batch_recommendation_processess=1,
                                                  entropy_bonus=0.0,
                                                  ppo_lr=1e-4,
                                                  use_klpen=True
                                                  )
        with open(CHECKPOINT + "/data_stats.json") as f:
            data_stats = json.load(f)        
        recommender.tune = lambda: None
        return recommender

recommenders = {
} 

def vanilla_sasrec(loss='bce', num_samples=1, batch_size=128):
    from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(vanilla=True, embedding_size=256, loss=loss, vanilla_num_negatives=num_samples)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=batch_size)

def sasrec_style_model(model_config, sequence_splitter, 
                target_builder,
                max_epochs=10000, 
                batch_size=128,
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs,
                                early_stop_epochs=200,
                                batch_size=batch_size,
                                eval_batch_size=256, #no need for gradients, should work ok
                                validation_batch_size=256,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)

def get_bert_style_model(model_config, tuning_samples_portion, batch_size=128):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        from tensorflow.keras.optimizers import Adam
        recommender_config = SequentialRecommenderConfig(model_config, 
                                               train_epochs=10000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               eval_batch_size=256, #no need for gradients, should work ok
                                               validation_batch_size=256,
                                               sequence_splitter=lambda: ItemsMasking(tuning_samples_prob=tuning_samples_portion), 
                                               max_batches_per_epoch=batch_size,
                                               targets_builder=ItemsMaskingTargetsBuilder,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               use_keras_training=True,
                                               optimizer=Adam(learning_rate=1e-4),
                                               sequence_length=SEQUENCE_LENGTH)
        
        return SequentialRecommender(recommender_config)

def full_bert(loss='softmax_ce', tuning_samples_portion=0.0):
        from aprec.recommenders.sequential.models.bert4rec.full_bert import FullBERTConfig
        model_config =  FullBERTConfig(embedding_size=256, loss=loss)
        return get_bert_style_model(model_config, tuning_samples_portion=tuning_samples_portion, batch_size=128)

def mf_bpr():
        return LightFMRecommender(num_latent_components=256, num_threads=32)


def gsasrec(num_samples=256, t=0.75):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(vanilla=True, embedding_size=256, loss='bce', vanilla_num_negatives=num_samples, 
                                vanilla_bce_t=t)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=128)



#recommenders["bert4rec"] = full_bert 
#recommenders["top"] = TopRecommender
#recommenders["mf_bpr"] = mf_bpr
recommenders[f"gptrec_supervised_checkpoint"] = lambda: generative_tuning_recommender(checkpoint_dir = CHECKPOINT)
#recommenders["vanilla_sasrec"] = vanilla_sasrec

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
