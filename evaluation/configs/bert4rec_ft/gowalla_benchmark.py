
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_ft.common_benchmark_config import *

DATASET = "gowalla_warm5"
N_VAL_USERS=1024
MAX_TEST_USERS=8196
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=False)