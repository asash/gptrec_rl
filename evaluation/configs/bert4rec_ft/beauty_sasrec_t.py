from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_ft.common_benchmark_config_sasrec_t import *

DATASET = "BERT4rec.beauty"
N_VAL_USERS=2048
MAX_TEST_USERS=8196
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)