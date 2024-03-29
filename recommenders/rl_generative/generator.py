import os
import numpy as np
import tensorflow as tf
from aprec.recommenders.rl_generative.utils import get_latest_checkpoint

from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecModel


class Generator(object):
    def __init__(self, model_config, model_checkpoint_dir, items, pred_history_vectorizer, gen_limit=10):
        self.model = RLGPT2RecModel.from_config(model_config)
        self.items = items
        self.pred_history_vectorizer = pred_history_vectorizer 
        self.gen_limit = gen_limit
        self.last_update_timestamp = None
        self.model_checkpoint_path = model_checkpoint_dir

    def try_update_weights(self):
        latest_checkpoint = get_latest_checkpoint(self.model_checkpoint_path)
        file_timestamp = os.path.getmtime(latest_checkpoint + "/__success__")
        if self.last_update_timestamp is None or file_timestamp > self.last_update_timestamp:
            self.last_update_timestamp = file_timestamp
            self.model.load_weights(latest_checkpoint + "/model.h5")
            print("Model weights updated")
        self.model.fit_biases(None) #fix this later
        
    def generate(self, input_seq, filter_seen, sep_item_id, greedy=False, train=False):
        self.try_update_weights()
        items = self.items
        gen_limit = self.gen_limit
        pred_history_vectorizer = self.pred_history_vectorizer
        model = self.model
        return static_generate(input_seq, filter_seen, sep_item_id, greedy, train, items, gen_limit, pred_history_vectorizer, model) 

def static_generate(input_seq, filter_seen, sep_item_id, greedy, train, items, gen_limit, pred_history_vectorizer, model):
    model_actions = [(0, action) for action in input_seq]
    mask = np.zeros([model.tokenizer.vocab_size+1], dtype='float32')
    mask[sep_item_id] = 1.0
    if filter_seen:
        for i in range (len(model_actions)):
            mask[model_actions[i][1]] = 1.0
    mask[items.size():] = 1.0
    masks = []
    generated_tokens = []
    position_ids = None 
    next_token_probs = []
    all_full_probs = []
    for i in range(gen_limit):
        seq = pred_history_vectorizer(model_actions, extension = i+1) 
        position_ids = shift_position_ids(position_ids, seq, model.data_parameters.sequence_length)
        tokens = model.tokenizer(seq, 1, model.data_parameters.sequence_length + i+1)
        attention_mask = tf.cast((tokens != -100), 'float32')
        result = model.gpt(tf.nn.relu(tokens[0]), attention_mask=attention_mask, training=train, position_ids=position_ids)
        next_token_logits = result.logits[-1, :] 
        past_key_values = result.past_key_values
        mask_score =  -1e6 
        masked_logits = tf.where(mask, mask_score, next_token_logits) 
        masks.append(mask.copy())
        if not greedy:
            next_token = sep_item_id 
            while next_token >= sep_item_id: #we don't want to generate SEP token or any other special tokens. Usually, this loop should only run once
                next_token = tf.random.categorical(tf.expand_dims(masked_logits, 0), num_samples=1)[-1,0].numpy()
                full_probs = tf.nn.softmax(tf.expand_dims(masked_logits, 0))[0].numpy()
                next_token_prob = full_probs[next_token] 
                all_full_probs.append(full_probs)
                next_token_probs.append(next_token_prob)
        else:
            next_token = tf.argmax(masked_logits[:sep_item_id]).numpy()
        model_actions.append((i+1, next_token))
        generated_tokens.append(next_token)
        if filter_seen:
            mask[next_token] = 1.0
    return generated_tokens, seq, next_token_probs, masks, all_full_probs

def shift_position_ids(position_ids, seq, sequence_length):
    if position_ids is None:
        position_ids =  np.arange(len(seq) -1, -1, -1)
    else:
        if position_ids[-1] == 0:
            position_ids = np.concatenate([position_ids, [sequence_length + 1]])
            pass
        else:
            position_ids = np.concatenate([position_ids, [position_ids[-1]+1]]) 
    return position_ids