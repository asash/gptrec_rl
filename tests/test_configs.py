import importlib.util
import os
import unittest
from aprec.datasets.datasets_register import DatasetsRegister

from aprec.utils.os_utils import get_dir, recursive_listdir
from aprec.evaluation.split_actions import ActionsSplitter
from aprec.evaluation.metrics.metric import Metric
from aprec.recommenders.recommender import Recommender


class TestConfigs(unittest.TestCase):
    def test_configs(self):
        configs_dir = os.path.join(os.path.join(get_dir()), "evaluation/configs")
        for filename in recursive_listdir(configs_dir):
            if self.should_ignore(filename):
                continue
            self.validate_config(filename)

    def should_ignore(self, filename):
        if not (filename.endswith(".py")):
            return True
        if "/common/" in filename:
            return True
        if "__pycache__" in filename:
            return True
        if "__init__" in filename:
            return True
        return False

    def validate_config(self, filename):
        print(f"validating {filename}")
        config_name = os.path.basename(filename[:-3])
        spec = importlib.util.spec_from_file_location(config_name, filename)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        required_fields = ["DATASET", "METRICS", "SPLIT_STRATEGY", "RECOMMENDERS", "USERS_FRACTIONS"]
        for field in required_fields:
            self.assertTrue(hasattr(config, field), f"missing required field {field}")

        self.assertTrue(config.DATASET in DatasetsRegister().all_datasets(), f"Unknown dataset {config.DATASET}")
        self.assertTrue(isinstance(config.SPLIT_STRATEGY, ActionsSplitter), f"Split strategy has wrong type: f{type(config.SPLIT_STRATEGY)}")
        self.assertTrue(len(config.METRICS) > 0)
        for metric in config.METRICS:
            self.assertTrue(isinstance(metric, Metric))

        for fraction in config.USERS_FRACTIONS:
            self.assertTrue(isinstance(fraction, (float, int)))
            self.assertGreater(fraction, 0)
            self.assertLessEqual(fraction, 1)

        for recommender_name in config.RECOMMENDERS:
            recommender = config.RECOMMENDERS[recommender_name]()
            self.assertTrue(isinstance(recommender, Recommender), f"bad recommender type of {recommender_name}")
            del(recommender)

        if hasattr(config, "USERS"):
            self.assertTrue(callable(config.USERS), "USERS should be callable")

