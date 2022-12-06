from aprec.evaluation.split_actions import LeaveOneOut

from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


USERS_FRACTIONS = [1.0]


def dnn(model_config, loss, sequence_splitter, 
                target_builder,
                pred_history_vectorizer,
                training_time_limit=3600,  
                max_epochs=10000, 
                sequence_length=50
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs, loss=loss,
                                early_stop_epochs=max_epochs,
                                batch_size=128,
                                max_batches_per_epoch=256,
                                training_time_limit=training_time_limit,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                pred_history_vectorizer=pred_history_vectorizer,
                                sequence_length=sequence_length
                                )
    
    return SequentialRecommender(config)

def sasrec_rss(recency_importance, add_cls=False, pos_smoothing=0,
               pos_embedding='default', pos_embeddding_comb='add', 
               causal_attention = True):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig

        from aprec.recommenders.sequential.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance
        from aprec.losses.bce import BCELoss


        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance), add_cls=add_cls)
        pred_history_vectorizer = AddMaskHistoryVectorizer() if add_cls else DefaultHistoryVectrizer()
        model_config = SASRecConfig(vanilla=False, num_heads=1, 
                   pos_embedding=pos_embedding,
                   pos_emb_comb=pos_embeddding_comb,
                   pos_smoothing=pos_smoothing, 
                   causal_attention=causal_attention) 

        return dnn(
            model_config,
            BCELoss(),
            sequence_splitter=target_splitter,
            target_builder=FullMatrixTargetsBuilder, 
            pred_history_vectorizer=pred_history_vectorizer)

def vanilla_sasrec():
    from aprec.losses.bce import BCELoss
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer

    sequence_length = 50
    model_config = SASRecConfig(vanilla=True )

    return dnn(model_config,  BCELoss(),
            ShiftedSequenceSplitter,
            target_builder=lambda: NegativePerPositiveTargetBuilder(sequence_length),
            sequence_length=sequence_length, 
            pred_history_vectorizer= DefaultHistoryVectrizer())




recommenders = {
#    "Sasrec-rss-lambdarank-0.8-nocls-exp-mult-sm8": lambda: sasrec_rss(0.8, add_cls=False, pos_smoothing=8, pos_embedding='exp', pos_embeddding_comb='mult'),
#    "Sasrec-rss-lambdarank-0.8-default-bidirectional": lambda: sasrec_rss(0.8, pos_embeddding_comb='add', pos_embedding='default', causal_attention=False),
#    "Sasrec-rss-lambdarank-0.8-default-causal": lambda: sasrec_rss(0.8, pos_embeddding_comb='add', pos_embedding='default', causal_attention=True),
    "Sasrec-rss-vanilla": lambda: vanilla_sasrec(),
    # "Sasrec-rss-lambdarank-0.8-cls-exp-mult": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='exp', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-cls-sin-mult": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='sin', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-nocls-sin-mult": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='sin', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-cls-default-mult": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='default', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-nocls-default-mult": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='default', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-cls-exp-add": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='exp', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-nocls-exp-add": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='exp', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-cls-sin-add": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='sin', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-nocls-sin-add": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='sin', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-cls-default-add": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='default', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-nocls-default-add": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='default', pos_embeddding_comb='add'),
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

def get_recommenders(filter_seen: bool, filter_recommenders = set()):
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
                continue
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

DATASET = "yelp_warm5"
N_VAL_USERS=1024
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":

    from aprec.tests.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)