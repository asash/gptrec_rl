import random
from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsWithReplacementSampler
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [NDCG(10), HighestScore(), NDCG(40), HIT(1), MRR(), 
                     Confidence('Softmax'), Confidence('Sigmoid'), Entropy('Softmax', 10)]

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200
EMBEDDING_SIZE=128
 



def vanilla_sasrec(alpha: float, num_negatives: int, embeddings_norm: float):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(vanilla=True, embedding_size=EMBEDDING_SIZE,
                                vanilla_num_negatives=num_negatives, vanilla_positive_alpha=alpha, 
                                embeddings_l2=embeddings_norm)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=128)


def sasrec_style_model(model_config, sequence_splitter, 
                target_builder,
                max_epochs=10000, 
                batch_size=1024,
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs,
                                early_stop_epochs=200,
                                batch_size=batch_size,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)




recommenders = {

        }

alphas = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 
              0.00390625, 
              0.001953125, 
              0.0009765625, 
              0.00048828125, 
              0.000244140625, 
              0.0001220703125]
negative_nums = [1, 2, 4, 6, 8, 16, 32, 64, 128]
embedding_norms = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.0]


for i in range(1000):
    alpha = random.choice(alphas)
    num_negatives = random.choice(negative_nums)
    l2 = random.choice(embedding_norms)
    recommenders[f"SASRec-alpha:{alpha}:negatives:{num_negatives}:embedding_norms:{l2}"] =\
        lambda a=alpha, n=num_negatives, norm=l2: vanilla_sasrec(alpha=a, num_negatives=n, embeddings_norm=norm)



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

