from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.common.common_benhmark_config import *

DATASET = "ml-100k"
N_VAL_USERS=128
MAX_TEST_USERS=943
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)