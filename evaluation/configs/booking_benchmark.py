import random

from aprec.datasets.booking import get_booking_dataset
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = get_booking_dataset(unix_timestamps=True)[0] 

USERS_FRACTIONS = [1.]

def top_recommender():
    return TopRecommender()

def svd_recommender(k):
    return SvdRecommender(k)

def dnn_sequential(model_arch, loss,
           session_len=50,
           ndcg_at = 50,
           activation_override=None,
           lambdas_normalization = True,
           learning_rate=0.001,
           num_main_layers=2,
           num_dense_layers=1,
           loss_pred_truncate=None,
           loss_bce_weight=0.0,
           log_lambdas=False,
           train_on_last=False,
           caser_uid = True,
           ):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                    model_arch=model_arch,
                                    optimizer=Adam(learning_rate),
                                    early_stop_epochs=10000,
                                    batch_size=128, sigma=1.0, ndcg_at=ndcg_at,
                                    max_history_len=session_len,
                                    output_layer_activation=activation,
                                    training_time_limit = 3600,
                                    eval_ndcg_at=40,
                                    target_decay=1.0,
                                    loss_lambda_normalization=lambdas_normalization,
                                    loss_pred_truncate=loss_pred_truncate,
                                    loss_bce_weight=loss_bce_weight,
                                    log_lambdas_len=log_lambdas,
                                    num_main_layers=num_main_layers,
                                    num_dense_layers=num_dense_layers,
                                    train_on_last_item_only=train_on_last,
                                    caser_use_user_id=caser_uid
                                    )

recommenders_raw = {
    #GRU-default
    "GRU4rec-BCE": lambda: dnn_sequential("gru", "binary_crossentropy"),
    "GRU4rec-Lambdarank-Vanilla": lambda: dnn_sequential("gru", "lambdarank"),
    "GRU4rec-Lambdarank-Truncated:2500": lambda: dnn_sequential("gru", "lambdarank", loss_pred_truncate=2500),
    "GRU4rec-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn_sequential("gru", "lambdarank", loss_pred_truncate=2500, loss_bce_weight=0.975),

    #GRU-train_on_last
    "GRU4rec-lastOnly-BCE": lambda: dnn_sequential("gru", "binary_crossentropy", train_on_last=True),
    "GRU4rec-lastOnly-Lambdarank-Vanilla": lambda: dnn_sequential("gru", "lambdarank", train_on_last=True),
    "GRU4rec-lastOnly-Lambdarank-Truncated:2500-lastOnly": lambda: dnn_sequential("gru", "lambdarank", loss_pred_truncate=2500, train_on_last=True),
    "GRU4rec-lastOnly-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn_sequential("gru", "lambdarank",
                                                                                 loss_pred_truncate=2500,
                                                                                 loss_bce_weight=0.975, train_on_last=True),

    # CASER-default
    "CASER-BCE": lambda: dnn_sequential("caser", "binary_crossentropy"),
    "CASER-Lambdarank-Vanilla": lambda: dnn_sequential("caser", "lambdarank"),
    "CASER-Lambdarank-Truncated:2500": lambda: dnn_sequential("caser", "lambdarank", loss_pred_truncate=2500),
    "CASER-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn_sequential("caser", "lambdarank",
                                                                                 loss_pred_truncate=2500,
                                                                                 loss_bce_weight=0.975),

    # CASER-train_on_last
    "CASER-lastOnly-BCE": lambda: dnn_sequential("caser", "binary_crossentropy", train_on_last=True),
    "CASER-lastOnly-Lambdarank-Vanilla": lambda: dnn_sequential("caser", "lambdarank", train_on_last=True),
    "CASER-lastOnly-Lambdarank-Truncated:2500-lastOnly": lambda: dnn_sequential("caser", "lambdarank",
                                                                                  loss_pred_truncate=2500,
                                                                                  train_on_last=True),
    "CASER-lastOnly-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn_sequential("caser", "lambdarank",
                                                                                          loss_pred_truncate=2500,
                                                                                          loss_bce_weight=0.975,
                                                                                          train_on_last=True),

    # CASER-nouid-nouid-default
    "CASER-nouid-BCE": lambda: dnn_sequential("caser", "binary_crossentropy", caser_uid=False),
    "CASER-nouid-Lambdarank-Vanilla": lambda: dnn_sequential("caser", "lambdarank", caser_uid=False),
    "CASER-nouid-Lambdarank-Truncated:2500": lambda: dnn_sequential("caser", "lambdarank", loss_pred_truncate=2500, caser_uid=False),
    "CASER-nouid-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn_sequential("caser", "lambdarank",
                                                                               loss_pred_truncate=2500,
                                                                               loss_bce_weight=0.975, caser_uid=False),

    # CASER-nouid-nouid-train_on_last
    "CASER-nouid-lastOnly-BCE": lambda: dnn_sequential("caser", "binary_crossentropy", train_on_last=True, caser_uid=False),
    "CASER-nouid-lastOnly-Lambdarank-Vanilla": lambda: dnn_sequential("caser", "lambdarank", train_on_last=True, caser_uid=False),
    "CASER-nouid-lastOnly-Lambdarank-Truncated:2500-lastOnly": lambda: dnn_sequential("caser", "lambdarank",
                                                                                loss_pred_truncate=2500,
                                                                                train_on_last=True, caser_uid=False),
    "CASER-nouid-lastOnly-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn_sequential("caser", "lambdarank",
                                                                                        loss_pred_truncate=2500,
                                                                                        loss_bce_weight=0.975,
                                                                                        train_on_last=True, caser_uid=False),

}

all_recommenders = list(recommenders_raw.keys())
random.shuffle(all_recommenders)


RECOMMENDERS = {
        "svd_recommender": lambda: svd_recommender(30), 
        "top_recommender": top_recommender, 

    }
for model in all_recommenders:
    RECOMMENDERS[model] = recommenders_raw[model]

print(f"evaluating {len(RECOMMENDERS)} models")

N_VAL_USERS=1024
MAX_TEST_USERS=32768

METRICS = [NDCG(10), NDCG(2), NDCG(5), NDCG(20), NDCG(40), Precision(10), Recall(10), HIT(1), HIT(10), MRR(), MAP(10)]


SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)