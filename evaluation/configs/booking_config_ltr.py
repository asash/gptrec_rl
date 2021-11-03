import copy
import os

from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.booking_recommender.booking_recommender_ltr import BookingRecommenderLTR
from aprec.evaluation.metrics.sps import SPS
from aprec.datasets.booking import get_booking_dataset
from tqdm import tqdm


DATASET, SUBMIT_ACTIONS = get_booking_dataset('./booking_data/booking_train_set.csv',
                              './booking_data/booking_test_set.csv')

GENERATE_SUBMIT_THRESHOLD = 0.54175

def generate_submit(recommender, recommender_name, evaluation_result, config):
    if evaluation_result["SPS@4"] < config.GENERATE_SUBMIT_THRESHOLD:
        print("SPS@4 is less than threshold, not generating the submit")
        return

    print("generating submit...")
    with open(os.path.join(config.out_dir, recommender_name + "_submit_" + ".csv"), 'w') as out_file:
        out_file.write("utrip_id,city_id_1,city_id_2,city_id_3,city_id_4\n")
        for action in tqdm(config.SUBMIT_ACTIONS, ascii=True):
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


def LTR(model_type, attention, lgbm_objecitve='lambdarank', lgbm_boosting_type='gbdt'):
    return BookingRecommenderLTR(batch_size=250, n_val_users=4000,
                                 candidates_cnt=500, val_epoch_size=4000, epoch_size=10000,
                                 num_training_samples=5000000, model_type=model_type, attention=attention,
                                 lgbm_objective=lgbm_objecitve, lgbm_boosting_type=lgbm_boosting_type)

RECOMMENDERS = {}

RECOMMENDERS["APREC-Neural-NeuralLambdarank"] =  lambda: LTR('neural', False)
RECOMMENDERS["APREC-Neural_attention-NeuralLambdarank"] =  lambda: LTR('neural', True)

for boosting_type in ('rf', 'gbdt', 'dart'):
    for objective in ('regression', 'lambdarank', 'rank_xendcg', 'regression_l1', 'huber',
                      'fair', 'poisson', 'quantile', 'mape', 'tweedie'):
        RECOMMENDERS[f"Lightgbm-{boosting_type}-{objective}"] = lambda boosting_type=boosting_type, objective=objective: LTR('lightgbm', False,
                                                             lgbm_boosting_type=boosting_type, lgbm_objecitve=objective)

SPLIT_STRATEGY = "LEAVE_ONE_OUT"
USERS_FRACTIONS = [1.0]
METRICS = [Precision(4), SPS(4), NDCG(4), NDCG(40)]
