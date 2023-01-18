from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_ft.common_benchmark_config import *

DATASET = "BERT4rec.beauty"
N_VAL_USERS=512
MAX_TEST_USERS=40226
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)