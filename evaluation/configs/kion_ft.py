from aprec.recommenders.sequential.target_builders.negative_samplers import SVDSimilaritySampler
from aprec.datasets.mts_kion import get_submission_user_ids, get_users, get_items
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from tqdm import tqdm

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

DATASET = "mts_kion"
USERS = get_users
ITEMS = get_items

GENERATE_SUBMIT_THRESHOLD =  0.0

def generate_submit(recommender, recommender_name, evaluation_result, config):
    submit_user_ids = get_submission_user_ids()
    if evaluation_result["MAP@10"] <= config.GENERATE_SUBMIT_THRESHOLD:
        print("MAP@10 is less than threshold, not generating the submit")
        return

    print("generating submit...")
    with open(os.path.join(config.out_dir, recommender_name + "_submit_" + ".csv"), 'w') as out_file:
        out_file.write("user_id,item_id\n")
        for user_id in tqdm(submit_user_ids, ascii=True):
            recommendations = recommender.recommend(user_id, limit=10)
            content_ids = [recommendation[0] for recommendation in recommendations]
            line = user_id + ",\"["  +  ", ".join(content_ids) + "]\"\n"
            out_file.write(line)

CALLBACKS = (generate_submit, )

def bert4rec_ft(negatives_sampler=SVDSimilaritySampler(sample_size=400)):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.sequential.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder
        from aprec.recommenders.sequential.models.bert4recft.bert4recft import BERT4RecFT
        from aprec.recommenders.sequential.target_builders.negative_samplers import RandomNegativesSampler
        from aprec.losses.bce import BCELoss
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
        sequence_len = 200
        model = BERT4RecFT(max_history_len=sequence_len)
        batch_size = 256 
        negatives_per_positive = negatives_sampler.get_sample_size()
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               training_time_limit=3600000, 
                                               loss = ItemsMaksingLossProxy(BCELoss(), negatives_per_positive, sequence_len),
                                               sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingWithNegativesTargetsBuilder(negatives_sampler=RandomNegativesSampler(negatives_per_positive)),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=24
                                               )
        return recommender

N_VAL_USERS=1024
MAX_TEST_USERS=4096

METRICS = [MAP(10), NDCG(10), NDCG(2), NDCG(5), NDCG(20), NDCG(40), Precision(10), Recall(10), HIT(1), HIT(10), MRR()]


RECOMMENDERS = {
        #"top_recommender": lambda: TopRecommender(0.01),
        #"MF-BPR": lambda: LightFMRecommender(256)
        "bert4rec_ft": lambda: FilterSeenRecommender(bert4rec_ft())
    }

USERS_FRACTIONS = [1.0]
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS, remove_single_action=False, recently_interacted_hours=7*24)
FILTER_COLD_START=False

