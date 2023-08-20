from aprec.datasets.bert4rec_datasets import get_movielens1m_genres
from aprec.datasets.datasets_register import DatasetsRegister
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.pcount import PCOUNT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.greedy_pcount_recommender import PCountRecommender


USERS_FRACTIONS = [1.0]
DATASET = "BERT4rec.ml-1m"
RECOMMENDATIONS_LIMIT=10
PCOUNT_CUTOFF=1000
SAVE_MODELS=False

genre_func = get_movielens1m_genres
METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()), PCOUNT(10, DatasetsRegister()[DATASET]()) ]
actions = DatasetsRegister()[DATASET]()

BERT4REC_CHECKPOINT = "/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/ml1m_baselines/bert4rec.dill.gz"

RECOMMENDERS={
    "BERT4Rec-PCOUNT-0.0": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.0),
    "BERT4Rec-PCOUNT-0.1": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.1),
    "BERT4Rec-PCOUNT-0.2": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.2),
    "BERT4Rec-PCOUNT-0.3": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.3),
    "BERT4Rec-PCOUNT-0.4": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.4),
    "BERT4Rec-PCOUNT-0.5": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.5),
    "BERT4Rec-PCOUNT-0.6": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.6),
    "BERT4Rec-PCOUNT-0.7": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.7),
    "BERT4Rec-PCOUNT-0.8": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.8),
    "BERT4Rec-PCOUNT-0.9": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 0.9),
    "BERT4Rec-PCOUNT-1.0": lambda: PCountRecommender(BERT4REC_CHECKPOINT, actions,  PCOUNT_CUTOFF, 1.0),
}

N_VAL_USERS=512
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)


if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)

