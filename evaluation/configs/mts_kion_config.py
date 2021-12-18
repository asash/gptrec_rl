import os
import random

from tqdm import tqdm

from aprec.datasets.mts_kion import get_mts_kion_dataset, get_submission_user_ids
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.sps import SPS
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.recommenders.deep_mf import DeepMFRecommender

DATASET = get_mts_kion_dataset()
SUBMIT_USER_IDS = get_submission_user_ids()

GENERATE_SUBMIT_THRESHOLD = 0.10

def generate_submit(recommender, recommender_name, evaluation_result, config):
    if evaluation_result["MAP@10"] <= config.GENERATE_SUBMIT_THRESHOLD:
        print("SPS@4 is less than threshold, not generating the submit")
        return

    print("generating submit...")
    with open(os.path.join(config.out_dir, recommender_name + "_submit_" + ".csv"), 'w') as out_file:
        out_file.write("user_id,item_id\n")
        for user_id in tqdm(config.SUBMIT_USER_IDS, ascii=True):
            recommendations = recommender.get_next_items(user_id, limit=10)
            content_ids = [recommendation[0] for recommendation in recommendations]
            line = user_id + ",\"["  +  ", ".join(content_ids) + "]\"\n"
            out_file.write(line)

CALLBACKS = (generate_submit, )

def deepmf(users_per_sample, items_per_sample, loss, truncation_level=None, bce_weight=0.0):
    if loss == 'lambdarank':
        loss = LambdaGammaRankLoss(items_per_sample, users_per_sample,
                                   ndcg_at=50, pred_truncate_at=truncation_level,
                                   bce_grad_weight=bce_weight, remove_batch_dim=True)
    return FilterSeenRecommender(DeepMFRecommender(users_per_sample, items_per_sample, loss, steps=1500))

USERS_FRACTIONS = [1.]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def vanilla_bert4rec(num_steps):
    max_seq_length = 50
    masked_lm_prob = 0.2
    max_predictions_per_seq = 20
    batch_size = 256
    num_train_steps = num_steps

    prop_sliding_window = 0.5
    mask_prob = 1.0
    dupe_factor = 10
    pool_size = 10

    num_warmup_steps = 100
    learning_rate = 1e-4
    random_seed = 31337

    bert_config = {
        "attention_probs_dropout_prob": 0.2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.2,
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "max_position_embeddings": 200,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "type_vocab_size": 2,
    }
    recommender = VanillaBERT4Rec(max_seq_length=max_seq_length,
                                  masked_lm_prob=masked_lm_prob,
                                  max_predictions_per_seq=max_predictions_per_seq,
                                  mask_prob=mask_prob,
                                  dupe_factor=dupe_factor,
                                  pool_size=pool_size,
                                  prop_sliding_window=prop_sliding_window,
                                  random_seed=random_seed,
                                  bert_config=bert_config,
                                  num_warmup_steps=num_warmup_steps,
                                  num_train_steps=num_train_steps,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate)
    return FilterSeenRecommender(recommender)

def salrec(loss, num_blocks, learning_rate, ndcg_at,
                session_len,  lambdas_normalization, activation_override=None,
                loss_pred_truncate=None,
                loss_bce_weight=0.0,
                log_lambdas=False
           ):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return FilterSeenRecommender(SalrecRecommender(train_epochs=10000, loss=loss,
                                                   optimizer=Adam(learning_rate),
                                                   early_stop_epochs=10000,
                                                   batch_size=128, sigma=1.0, ndcg_at=ndcg_at,
                                                   max_history_len=session_len,
                                                   output_layer_activation=activation,
                                                   training_time_limit = 3600*12,
                                                   num_blocks=num_blocks,
                                                   num_target_predictions=5,
                                                   eval_ndcg_at=40,
                                                   target_decay=0.8,
                                                   loss_lambda_normalization=lambdas_normalization,
                                                   loss_pred_truncate=loss_pred_truncate,
                                                   loss_bce_weight=loss_bce_weight,
                                                   log_lambdas_len=log_lambdas
                                                   ))

def mlp_historical_embedding(loss, activation_override=None):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return FilterSeenRecommender(DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                                          optimizer=Adam(), early_stop_epochs=100,
                                                          batch_size=64, sigma=1.0, ndcg_at=40,
                                                          bottleneck_size=64,
                                                          max_history_len=150,
                                                          output_layer_activation=activation, target_decay=0.8))

recommenders_raw = {
    "Transformer-Lambdarank-blocks:3-lr:0.001-ndcg:50-session_len:100-lambda_norm:True-truncate:2500-bce_weight:0.975":
        lambda: salrec('lambdarank', 3, 0.001, 50, 10, True, loss_pred_truncate=2500, loss_bce_weight=0.975),
}

all_recommenders = list(recommenders_raw.keys())
random.shuffle(all_recommenders)


RECOMMENDERS = {
#        "svd_recommender": lambda: svd_recommender(30),
        "top_recommender": top_recommender,

    }
for model in all_recommenders:
    RECOMMENDERS[model] = recommenders_raw[model]

print(f"evaluating {len(RECOMMENDERS)} models")

N_VAL_USERS=256
MAX_TEST_USERS=4096

METRICS = [NDCG(10),  NDCG(2), NDCG(5), NDCG(20), NDCG(40), Precision(10), Recall(10), SPS(1), SPS(10), MRR(), MAP(10)]


SPLIT_STRATEGY = "LEAVE_ONE_OUT"
