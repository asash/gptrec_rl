from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.common_benhmark_config import *

DATASET = get_bert4rec_dataset("ml-1m")
N_VAL_USERS=256
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)