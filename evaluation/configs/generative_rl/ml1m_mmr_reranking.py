from aprec.datasets.bert4rec_datasets import get_movielens1m_genres
from aprec.datasets.datasets_register import DatasetsRegister
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.pcount import PCOUNT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.mmr_recommender import MMRRecommender


USERS_FRACTIONS = [1.0]
DATASET = "BERT4rec.ml-1m"
RECOMMENDATIONS_LIMIT=10
MMR_CUTOFF=1000
SAVE_MODELS=False

genre_func = get_movielens1m_genres
METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()), PCOUNT(10, DatasetsRegister()[DATASET]()) ]
BPR_CHECKPOINT = "./results/BERT4rec.ml-1m/ml1m_evaluate_checkpoint_2023_08_19T10_11_19/checkpoints/mf_bpr.dill.gz"

RECOMMENDERS={
    "MF-BPR-MMR-1.0": lambda: MMRRecommender(BPR_CHECKPOINT, genre_func(), MMR_CUTOFF, 1.0),
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

