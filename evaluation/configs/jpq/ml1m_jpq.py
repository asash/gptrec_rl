from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.jpq.common_benchmark_config_jpq import *

DATASET = "BERT4rec.ml-1m"
N_VAL_USERS=1024
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":

    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)