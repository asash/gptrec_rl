from aprec.evaluation.split_actions import LeaveOneOut
from evaluation.configs.common.common_benhmark_config import *

DATASET = "BERT4rec.ml-1m"
N_VAL_USERS=256
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)