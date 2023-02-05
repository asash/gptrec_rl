from __future__ import annotations
from typing import Type
import numpy as np
import tensorflow as tf
from aprec.losses import get_loss
from aprec.recommenders.sequential.models.positional_encodings import  get_pos_embedding
from aprec.recommenders.sequential.samplers.sampler import get_negatives_sampler
layers = tf.keras.layers


from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialModelConfig, SequentialRecsysModel
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec

class SASLiftModel(SequentialRecsysModel):
    @classmethod
    def get_model_config_class(cls) -> Type[SASLiftConfig]:
        return SASLiftConfig

    def fit_biases(self, train_users):
        cnt = np.ones(self.data_parameters.num_items)
        for user in train_users:
            user_actions = [action[1] for action in user] 
            for action in user_actions:
                cnt[action] += 1
        item_probs = cnt/np.sum(cnt)
        self.item_freqs.assign(item_probs) 
        self.item_idfs.assign(-np.log(item_probs))

    def __init__(self, model_parameters, data_parameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: SASLiftConfig #just a hint for the static analyser
        self.positions = tf.constant(tf.expand_dims(tf.range(self.data_parameters.sequence_length), 0))
        self.item_embeddings_layer = layers.Embedding(
                                                      self.data_parameters.num_items + 2, 
                                                      output_dim=self.model_parameters.embedding_size, dtype='float32', 
                                                      name='item_embeddings')
        if self.model_parameters.pos_emb_comb != 'ignore':
            self.postion_embedding_layer = get_pos_embedding(self.data_parameters.sequence_length, self.model_parameters.embedding_size, self.model_parameters.pos_embedding)
        self.embedding_dropout = layers.Dropout(self.model_parameters.dropout_rate, name='embedding_dropout')
    
        freqs_init = tf.random_uniform_initializer(0.00001, 0.1)
        self.item_freqs = tf.Variable(freqs_init(shape=(self.data_parameters.num_items,)), trainable=False) 

        idfs_init = tf.random_uniform_initializer(5, 9)
        self.item_idfs = tf.Variable(idfs_init(shape=(self.data_parameters.num_items,)), trainable=False) 

        self.attention_blocks = []
        for i in range(self.model_parameters.num_blocks):
            block_layers = {
                "first_norm": layers.LayerNormalization(),
                "attention_layers": {
                    "query_proj": layers.Dense(self.model_parameters.embedding_size, activation='linear'),
                    "key_proj": layers.Dense(self.model_parameters.embedding_size, activation='linear'),
                    "val_proj": layers.Dense(self.model_parameters.embedding_size, activation='linear'),
                    "dropout": layers.Dropout(self.model_parameters.dropout_rate),
                },
                "second_norm": layers.LayerNormalization(),
                "dense1": layers.Dense(self.model_parameters.embedding_size, activation='relu'),
                "dense2": layers.Dense(self.model_parameters.embedding_size),
                "dropout": layers.Dropout(self.model_parameters.dropout_rate)
            }
            self.attention_blocks.append(block_layers)
        self.output_activation = tf.keras.activations.get(self.model_parameters.output_layer_activation)
        self.seq_norm = layers.LayerNormalization()
        self.all_items = tf.range(0, self.data_parameters.num_items)
        if not self.model_parameters.reuse_item_embeddings:
            self.output_item_embeddings = layers.Embedding(self.data_parameters.num_items + 2, self.model_parameters.embedding_size)

        if self.model_parameters.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.model_parameters.embedding_size, activation='gelu')
        self.sampler = get_negatives_sampler(self.model_parameters.vanilla_target_sampler, 
                                                 self.data_parameters, self.model_parameters.vanilla_num_negatives)

    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x, attentions = multihead_attention(queries, keys, self.model_parameters.num_heads, self.attention_blocks[i]["attention_layers"],
                                     causality=self.model_parameters.causal_attention)
        x =x + queries
        x = self.attention_blocks[i]["second_norm"](x)
        residual = x
        x = self.attention_blocks[i]["dense1"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x = self.attention_blocks[i]["dense2"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x += residual
        x *= mask
        return x, attentions

    def get_dummy_inputs(self):
        pad = tf.cast(tf.fill((self.data_parameters.batch_size, 1), self.data_parameters.num_items), 'int64')
        seq = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length-1), 'int64')
        inputs = [tf.concat([pad, seq], -1)]
        positives = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int64')
        inputs.append(positives)
        return inputs

    def get_item_idfs(self):
        return tf.stop_gradient(self.item_idfs)

    def get_item_freqs(self):
        return tf.stop_gradient(self.item_freqs)


    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb, attentions = self.get_seq_embedding(input_ids, bs=self.data_parameters.batch_size, training=training)
        target_positives = tf.expand_dims(inputs[1], -1)
        target_negatives = self.sampler(input_ids, target_positives)
        target_ids = tf.concat([target_positives, target_negatives], -1)

        #use relu to hide negative targets - they will be ignored later in any case
        target_embeddings = self.get_target_embeddings(tf.nn.relu(target_ids))
        cnt_per_pos = self.model_parameters.vanilla_num_negatives + 1
        logits = tf.einsum("bse, bsne -> bsn", seq_emb, target_embeddings)
        #if self.model_parameters.vanilla_bce_t != 0:

        alpha = self.model_parameters.vanilla_num_negatives / (self.data_parameters.num_items - 1)
        t = self.model_parameters.vanilla_bce_t 
        beta = alpha * ((1 - 1/alpha)*t + 1/alpha)

        positive_scores = logits[:, :, 0:1]
        negative_scores = logits[:,:,1:]
        mask = tf.cast((input_ids != self.data_parameters.num_items), 'float32')
        positive_idfs = tf.gather(self.get_item_idfs(), tf.nn.relu(target_positives))
        negative_freqs = tf.gather(self.get_item_freqs(), tf.nn.relu(target_negatives))
        positive_logprobs = tf.squeeze(positive_scores - positive_idfs, -1)*beta
        negative_logprobs_sum = tf.reduce_sum(-tf.exp(negative_scores) * negative_freqs, axis=-1)
        loss_sum = tf.reduce_sum(-(positive_logprobs + negative_logprobs_sum) * mask, -1)
        num_unmasked = tf.reduce_sum(mask, -1)
        loss = tf.math.divide_no_nan(loss_sum, num_unmasked)
        return loss
    

    def score_all_items(self, inputs):
        input_ids = inputs[0]
        seq_emb, attentions = self.get_seq_embedding(input_ids)
        seq_emb = seq_emb[:, -1, :]
        target_ids = self.all_items
        target_embeddings = self.get_target_embeddings(target_ids)
        output = seq_emb @ tf.transpose(target_embeddings)
        output = tf.exp(output)*self.get_item_freqs()
        return output

    def get_target_embeddings(self, target_ids):
        if self.model_parameters.reuse_item_embeddings:
            target_embeddings = self.item_embeddings_layer(target_ids)
        else:
            target_embeddings = self.output_item_embeddings(target_ids)
        if self.model_parameters.encode_output_embeddings:
            target_embeddings = self.output_item_embeddings_encode(target_embeddings)
        return target_embeddings

    def get_seq_embedding(self, input_ids, bs=None, training=None):
        seq = self.item_embeddings_layer(input_ids)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.data_parameters.num_items), dtype=tf.float32), -1)
        if bs is None:
            bs = seq.shape[0]
        positions  = tf.tile(self.positions, [bs, 1])
        if training and self.model_parameters.pos_smoothing:
            smoothing = tf.random.normal(shape=positions.shape, mean=0, stddev=self.model_parameters.pos_smoothing)
            positions =  tf.maximum(0, smoothing + tf.cast(positions, 'float32'))
        if self.model_parameters.pos_emb_comb != 'ignore':
            pos_embeddings = self.postion_embedding_layer(positions)[:input_ids.shape[0]]
        if self.model_parameters.pos_emb_comb == 'add':
             seq += pos_embeddings
        elif self.model_parameters.pos_emb_comb == 'mult':
             seq *= pos_embeddings

        elif self.model_parameters.pos_emb_comb == 'ignore':
             seq = seq
        seq = self.embedding_dropout(seq)
        seq *= mask
        attentions = []
        for i in range(self.model_parameters.num_blocks):
            seq, attention = self.block(seq, mask, i)
            attentions.append(attention)
        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions 

class SASLiftConfig(SequentialModelConfig):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                dropout_rate=0.5, num_blocks=2, num_heads=1,
                reuse_item_embeddings=False,
                encode_output_embeddings=False,
                pos_embedding = 'learnable', 
                pos_emb_comb = 'add',
                causal_attention = True,
                pos_smoothing = 0,
                max_targets_per_user=10,
                vanilla_num_negatives = 1,
                vanilla_bce_t = 0.0,
                vanilla_target_sampler = 'random',
                full_target = False,
                ): 
        self.output_layer_activation=output_layer_activation
        self.embedding_size=embedding_size
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.reuse_item_embeddings = reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.pos_embedding = pos_embedding
        self.pos_emb_comb = pos_emb_comb
        self.causal_attention = causal_attention
        self.pos_smoothing = pos_smoothing
        self.max_targets_per_user = max_targets_per_user #only used with sparse positives
        self.full_target = full_target,
        self.vanilla_num_negatives = vanilla_num_negatives 
        self.vanilla_target_sampler = vanilla_target_sampler
        self.vanilla_bce_t = vanilla_bce_t

    def as_dict(self):
        result = self.__dict__
        return result
    
    def get_model_architecture(self) -> Type[SequentialRecsysModel]:
        return SASLiftModel


    
