import unittest

from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.utils.generator_limit import generator_limit


class TestVanillaBert4rec(unittest.TestCase):
    def test_vanilla_bert4rec(self):
        max_seq_length = 200
        masked_lm_prob = 0.2
        max_predictions_per_seq = 40
        mask_prob = 1.0
        dupe_factor = 10
        pool_size = 10
        prop_sliding_window = 0.5
        num_warmup_steps = 100
        num_train_steps = 200
        batch_size = 256
        learning_rate = 1e-4
        random_seed = 31337

        bert_config = {
            "attention_probs_dropout_prob": 0.2,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.2,
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "max_position_embeddings": 200,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "type_vocab_size": 2,
        }

        recommender = VanillaBERT4Rec(max_seq_length=max_seq_length,
                                      masked_lm_prob=masked_lm_prob,
                                      max_predictions_per_seq=max_predictions_per_seq,
                                      mask_prob=mask_prob,
                                      dupe_factor=dupe_factor,
                                      pool_size=pool_size,
                                      prop_sliding_window=prop_sliding_window,
                                      random_seed=random_seed,
                                      bert_config=bert_config,
                                      num_warmup_steps=num_warmup_steps,
                                      num_train_steps=num_train_steps,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate)
        for action in generator_limit(get_movielens20m_actions(), 100000):
            recommender.add_action(action)
        recommender.rebuild_model()
        print(recommender.get_next_items('120', 10))

if __name__ == "__main__":
    unittest.main()