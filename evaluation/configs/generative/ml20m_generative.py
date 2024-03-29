from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.generative.common_benchmark_config import *

DATASET = "ml-20m_warm5"
N_VAL_USERS=128
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":

    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)