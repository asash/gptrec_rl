from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.common.common_benchmark_bert4rec import *

DATASET = "BERT4rec.ml-1m"
N_VAL_USERS=2048
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)