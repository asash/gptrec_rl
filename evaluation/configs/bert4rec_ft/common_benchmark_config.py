import random
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import SVDSimilaritySampler
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]


def bert4rec_ft(negatives_sampler):
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder, RandomNegativesSampler
        from aprec.recommenders.dnn_sequential_recommender.models.bert4recft.bert4recft import BERT4RecFT
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import RandomNegativesSampler
        from aprec.losses.bce import BCELoss
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
        sequence_len = 100
        model = BERT4RecFT(max_history_len=sequence_len)
        batch_size = 256 
        negatives_per_positive = negatives_sampler.get_sample_size()
        metric = ItemsMaksingLossProxy(BCELoss(), negatives_per_positive, sequence_len)
        metric.set_batch_size(batch_size)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               training_time_limit=3600000, 
                                               loss = ItemsMaksingLossProxy(BCELoss(), negatives_per_positive, sequence_len),
                                               debug=False, sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingWithNegativesTargetsBuilder(relative_positions_encoding=True, 
                                                                                        negatives_sampler=RandomNegativesSampler(negatives_per_positive)),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=metric,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=24
                                               )
        return recommender


recommenders = {
    # "bert4rec_ft_svd_5_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=5, ann_sampling_factor=1)),
    # "bert4rec_ft_svd_5_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=5, ann_sampling_factor=2)),
    # "bert4rec_ft_svd_5_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=5, ann_sampling_factor=5)),
    # "bert4rec_ft_svd_5_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=5, ann_sampling_factor=10)),
    # "bert4rec_ft_svd_5_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=5, ann_sampling_factor=20)),

    # "bert4rec_ft_svd_1_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=1, ann_sampling_factor=1)),
    # "bert4rec_ft_svd_1_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=1, ann_sampling_factor=2)),
    # "bert4rec_ft_svd_1_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=1, ann_sampling_factor=5)),
    # "bert4rec_ft_svd_1_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=1, ann_sampling_factor=10)),
    # "bert4rec_ft_svd_1_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=1, ann_sampling_factor=20)),

    # "bert4rec_ft_svd_2_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=2, ann_sampling_factor=1)),
    # "bert4rec_ft_svd_2_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=2, ann_sampling_factor=2)),
    # "bert4rec_ft_svd_2_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=2, ann_sampling_factor=5)),
    # "bert4rec_ft_svd_2_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=2, ann_sampling_factor=10)),
    # "bert4rec_ft_svd_2_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=2, ann_sampling_factor=20)),

    # "bert4rec_ft_svd_10_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=10, ann_sampling_factor=1)),
    # "bert4rec_ft_svd_10_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=10, ann_sampling_factor=2)),
    # "bert4rec_ft_svd_10_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=10, ann_sampling_factor=5)),
    # "bert4rec_ft_svd_10_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=10, ann_sampling_factor=10)),
    # "bert4rec_ft_svd_10_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=10, ann_sampling_factor=20)),

    # "bert4rec_ft_svd_20_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=20, ann_sampling_factor=1)),
    # "bert4rec_ft_svd_20_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=20, ann_sampling_factor=2)),
    # "bert4rec_ft_svd_20_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=20, ann_sampling_factor=5)),
    # "bert4rec_ft_svd_20_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=20, ann_sampling_factor=10)),
    # "bert4rec_ft_svd_20_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=20, ann_sampling_factor=20)),

    # "bert4rec_ft_svd_50_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=50, ann_sampling_factor=1)),
    # "bert4rec_ft_svd_50_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=50, ann_sampling_factor=2)),
    # "bert4rec_ft_svd_50_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=50, ann_sampling_factor=5)),
    # "bert4rec_ft_svd_50_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=50, ann_sampling_factor=10)),
    # "bert4rec_ft_svd_50_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=50, ann_sampling_factor=20)),

    "bert4rec_ft_svd_100_factor_1": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=100, ann_sampling_factor=1)),
    "bert4rec_ft_svd_100_factor_2": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=100, ann_sampling_factor=2)),
    "bert4rec_ft_svd_100_factor_5": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=100, ann_sampling_factor=5)),
    "bert4rec_ft_svd_100_factor_10": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=100, ann_sampling_factor=10)),
    "bert4rec_ft_svd_100_factor_20": lambda: bert4rec_ft(SVDSimilaritySampler(sample_size=100, ann_sampling_factor=20)),
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    all_recommenders = list(recommenders.keys())
    random.shuffle(all_recommenders)
    for recommender_name in all_recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

