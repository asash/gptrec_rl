import copy
import os

from aprec.evaluation.split_actions import get_control_users
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.conditional_top_recommender import ConditionalTopRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.item_item import ItemItemRecommender
from aprec.recommenders.transition_chain_recommender import TransitionsChainRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.booking_recommender.booking_recommender import BookingRecommender
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.sps import SPS
from aprec.datasets.booking import get_booking_dataset
from tqdm import tqdm


DATASET, SUBMIT_ACTIONS = get_booking_dataset('./booking_data/booking_train_set.csv',
                              './booking_data/booking_test_set.csv')

GENERATE_SUBMIT_THRESHOLD = 0.51

def generate_submit(recommender, recommender_name, evaluation_result, config):
    if evaluation_result["SPS@4"] < config.GENERATE_SUBMIT_THRESHOLD:
        print("SPS@4 is less than threshold, not generating the submit")
        return

    print("generating submit...")
    with open(os.path.join(config.out_dir, recommender_name + "_submit_" + ".csv"), 'w') as out_file:
        out_file.write("utrip_id,city_id_1,city_id_2,city_id_3,city_id_4\n")
        for action in tqdm(config.SUBMIT_ACTIONS):
            recommendations = recommender.get_next_items(action.user_id, limit=4, features=action)
            city_ids = [recommendation[0] for recommendation in recommendations]
            line = ",".join([action.user_id] + city_ids ) + "\n"
            out_file.write(line)


def features_from_test(test_actions):
    target_action = copy.deepcopy(test_actions[0])
    target_action.data['city_id'] = 0
    target_action.data['hotel_country'] = ''
    return target_action

FEATURES_FROM_TEST = features_from_test
CALLBACKS = (generate_submit, )

def top_recommender():
    return TopRecommender()

def conditional_top_recommender():
    return ConditionalTopRecommender('hotel_country')

def filter_seen_recommender(recommender):
    return FilterSeenRecommender(recommender)

def svd_recommender(k):
    return SvdRecommender(k)

def item_item_recommender():
    return ItemItemRecommender()

def mlp_historical_embedding(loss, activation_override=None, bottleneck_size=64, target_decay=0.6,
                                                           min_target_val = 0.03):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return BookingRecommender(train_epochs=10000, loss=loss,
                                        optimizer=Adam(), early_stop_epochs=30,
                                        batch_size=500, sigma=1.0, ndcg_at=40,
                                        n_val_users=4000,
                                        bottleneck_size=bottleneck_size,
                                        max_history_len=50,
                                        candidates_cnt=500,
                                        output_layer_activation=activation)

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "conditional_top_recommender": conditional_top_recommender,
    #"svd_recommender": lambda: svd_recommender(30),
#    "item_temem_recommender": item_item_recommender,
    "transitions_chain_recommender": TransitionsChainRecommender,
    "APREC-GMLPHE-Lambdarank-256-0.5": lambda: mlp_historical_embedding('lambdarank', 'linear', 256, target_decay=0.5),
}

SPLIT_STRATEGY = "LEAVE_ONE_OUT"
USERS_FRACTIONS = [1.0]
METRICS = [Precision(4), SPS(4), NDCG(4), NDCG(40)]
