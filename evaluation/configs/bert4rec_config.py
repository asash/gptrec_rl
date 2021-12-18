from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.sps import SPS
from datasets.bert4rec_datasets import get_bert4rec_dataset
from recommenders.filter_seen_recommender import FilterSeenRecommender
from recommenders.top_recommender import TopRecommender

DATASET = get_bert4rec_dataset("ml-1m")

USERS_FRACTIONS = [1]


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

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def top_recommender_no_filters():
    return TopRecommender()

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "top_recommender_no_filters": top_recommender_no_filters,
    "vanilla_bert4rec-400000": lambda: vanilla_bert4rec(400000),
    "vanilla_bert4rec-800000": lambda: vanilla_bert4rec(800000),
    "vanilla_bert4rec-1600000": lambda: vanilla_bert4rec(1600000),
    "vanilla_bert4rec-3200000": lambda: vanilla_bert4rec(1600000),
}


MAX_TEST_USERS=943

METRICS = [SPS(1), SPS(5), SPS(10), MRR()]


SPLIT_STRATEGY = "LEAVE_ONE_OUT"
