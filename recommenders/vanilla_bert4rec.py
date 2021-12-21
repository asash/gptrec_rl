#This code uses port of the original bert4rec implementation
import copy
import json
import os
import pickle
import random
import shlex
import subprocess
import tempfile
from collections import defaultdict
from BERT4rec.gen_data_fin import create_training_instances, write_instance_to_example_file
from BERT4rec.vocab import FreqVocab
import BERT4rec

from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId


class VanillaBERT4Rec(Recommender):
    def __init__(self, max_seq_length, dupe_factor, masked_lm_prob, max_predictions_per_seq, random_seed, mask_prob,
                 prop_sliding_window, pool_size, bert_config, batch_size, num_warmup_steps, num_train_steps,
                 learning_rate, training_time_limit=None):
        super().__init__()
        self.user_actions = defaultdict(list)
        self.user_ids = ItemId()
        self.item_ids = ItemId()
        self.max_seq_length = max_seq_length
        self.dupe_factor = dupe_factor
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.rng = random.Random(random_seed)
        self.mask_prob = mask_prob
        self.prop_sliding_window = prop_sliding_window
        self.pool_size = pool_size
        self.bert_config = bert_config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.training_time_limit = training_time_limit
        self.predictions_cache = {}

    def add_action(self, action):
        self.user_actions[action.user_id].append(action)

    def rebuild_model(self):
        pred_item = ["[PRED]"]
        bert4rec_docs = {}
        bert4rec_pred_docs = {}

        for user in self.user_actions:
            self.user_actions[user].sort(key=lambda action: action.timestamp)
            user_str, doc = self.get_bert4rec_doc(user)
            doc_for_prediction = doc + pred_item

            #mask_last code in the vanilla bert4rec implementation requires at least two docs in the collection
            if len(doc) > 1:
                bert4rec_docs[user_str] = doc

            bert4rec_pred_docs[user_str] = doc_for_prediction

        vocab = FreqVocab(bert4rec_pred_docs)
        user_test_data_output = {
            k: [vocab.convert_tokens_to_ids(v)]
            for k, v in bert4rec_pred_docs.items()
        }

        train_instances = create_training_instances(bert4rec_docs, self.max_seq_length, self.dupe_factor, self.masked_lm_prob,
                                              self.max_predictions_per_seq, self.rng, vocab, self.mask_prob,
                                              self.prop_sliding_window, self.pool_size, False)

        pred_instances = create_training_instances(bert4rec_pred_docs, self.max_seq_length, 1, self.masked_lm_prob,
                                              self.max_predictions_per_seq, self.rng, vocab, self.mask_prob,
                                              -1, self.pool_size, True)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_instances_filename = os.path.join(tmpdir, "train_instances.tfrecords")
            train_instances_file = open(train_instances_filename, "wb")

            pred_instances_filename = os.path.join(tmpdir, "pred_instances.tfrecords")
            pred_instances_file = open(pred_instances_filename, "wb")

            bert_config_filename = os.path.join(tmpdir, "bert_config_file.json")
            bert_config_file = open(bert_config_filename, "wb")

            vocab_filename = os.path.join(tmpdir, "vocab.pickle")
            vocab_file = open(vocab_filename, "wb")

            history_filename =  os.path.join(tmpdir, "history.pickle")
            history_file = open(history_filename, "wb")


            predictions_filename =  os.path.join(tmpdir, "predictions.csv")

            write_instance_to_example_file(train_instances,
                                       self.max_seq_length,
                                       self.max_predictions_per_seq, vocab, train_instances_file.name,)

            write_instance_to_example_file(pred_instances,
                                           self.max_seq_length,
                                           self.max_predictions_per_seq, vocab, pred_instances_file.name,)
            bert_config = copy.deepcopy(self.bert_config)
            bert_config["vocab_size"] = vocab.get_vocab_size()
            bert_config_file.write(json.dumps(bert_config, indent=4).encode("utf-8"))
            bert_config_file.flush()
            pickle.dump(vocab, vocab_file, protocol=2)
            pickle.dump(user_test_data_output, history_file, protocol=2)
            self.train_and_predict(train_instances_filename,
                                   pred_instances_filename,
                                   vocab_filename,
                                   history_filename,
                                   bert_config_filename,
                                   predictions_filename,
                                   tmpdir)
            pass

    def get_bert4rec_doc(self, user):
        user_id = self.user_ids.get_id(user)
        user_str = f"user_{user_id}"
        doc = [f"item_{self.item_ids.get_id(action.item_id)}" for action in self.user_actions[user]]
        return user_str, doc

    def train_and_predict(self, train_instances_filename,
                          pred_instances_filename,
                          vocab_filename, user_history_filename,
                          bert_config_filename,
                          predictions_filename,
                          tmpdir):
        bert4rec_dir = os.path.dirname(BERT4rec.__file__)
        bert4rec_runner = os.path.join(bert4rec_dir, "run.py")
        signature = tmpdir.split("/")[-1]
        cmd = f"python {bert4rec_runner}\
            --train_input_file={train_instances_filename} \
            --test_input_file={pred_instances_filename} \
            --vocab_filename={vocab_filename} \
            --user_history_filename={user_history_filename} \
            --checkpointDir={tmpdir} \
            --signature={signature}\
            --do_train=True \
            --do_eval=True \
            --bert_config_file={bert_config_filename} \
            --batch_size={self.batch_size} \
            --max_seq_length={self.max_seq_length} \
            --max_predictions_per_seq={self.max_predictions_per_seq} \
            --num_train_steps={self.num_train_steps} \
            --num_warmup_steps={self.num_warmup_steps} \
            --save_predictions_file={predictions_filename} \
            --learning_rate={self.learning_rate} "

        if self.training_time_limit is not None:
            cmd += f" --training-time-limit={self.training_time_limit}"

        subprocess.check_call(shlex.split(cmd))
        with open(predictions_filename) as predictions:
            for line in predictions:
                splits = line.strip().split(';')
                user_id = splits[0]
                self.predictions_cache[user_id] = []
                for item_with_score in  splits[1:]:
                    item, score = item_with_score.split(":")
                    score = float(score)
                    self.predictions_cache[user_id].append((item, score))

    def recommend(self, user_id, limit, features=None):
        internal_user_id = "user_" + str(self.user_ids.get_id(user_id))
        recs = self.predictions_cache[internal_user_id]
        result = []
        for internal_item_id, score in recs[:limit]:
            if not internal_item_id.startswith("item_"):
                continue
            item_id = self.item_ids.reverse_id(int(internal_item_id.split("_")[1]))
            result.append((item_id, score))
        return result

