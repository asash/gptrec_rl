from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.losses.mean_ypred_loss import MeanPredLoss
from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.sequential.models.bert4rec.bert4rec import BERT4Rec
from aprec.recommenders.sequential.models.albert4rec.albert4rec import ALBERT4Rec
from aprec.recommenders.sequential.models.mixer import RecsysMixer
from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from tensorflow.keras.optimizers import Adam

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def top_recommender():
    return TopRecommender()


def svd_recommender(k):
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    return LightFMRecommender(k, loss, num_threads=32)




def bert4rec(relative_position_encoding, sequence_len=50, rss = lambda n, k: 1, layers=2, arch=BERT4Rec, masking_prob=0.2):
        model = arch( max_history_len=sequence_len)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=128,
                                               training_time_limit=3600000, 
                                               loss = MeanPredLoss(),
                                               sequence_splitter=lambda: ItemsMasking(masking_prob=masking_prob, recency_importance=rss), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=relative_position_encoding),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               )
        return recommender
recommenders = {
    "mixer": lambda:bert4rec(True, 200, 
                arch=lambda max_history_len: RecsysMixer(max_history_len=max_history_len, num_blocks=6, 
                                                                                                       embedding_size=256, 
                                                                                                       token_mixer_hidden_dim=256, 
                                                                                                       channel_mixer_hidden_dim=256, 
                                                                                                       l2_emb=0.05),
                 layers=2, masking_prob=0.2), 
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

