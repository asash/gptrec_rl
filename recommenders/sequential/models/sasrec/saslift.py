from __future__ import annotations
from typing import Type
import numpy as np
import tensorflow as tf
from aprec.recommenders.sequential.models.positional_encodings import  get_pos_embedding

from aprec.recommenders.sequential.samplers.sampler import get_negatives_sampler
from scipy.sparse import csr_matrix
layers = tf.keras.layers

from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import KBinsDiscretizer

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec

class ItemCodeLayer(layers.Layer):
    def __init__(self, model_parameters: SASLiftConfig, data_parameters: SequentialDataParameters):
        super().__init__()
        self.model_parameters = model_parameters
        self.data_parameters = data_parameters
        self.sub_embedding_size = self.model_parameters.embedding_size // self.model_parameters.pq_m
        self.item_code_bytes = self.model_parameters.embedding_size // self.sub_embedding_size
        item_initializer = tf.zeros_initializer()
        self.item_codes = tf.Variable(item_initializer((self.data_parameters.num_items + 1, self.item_code_bytes), dtype='uint8'), trainable=False, name="ItemCodes/codes")

        centroid_initializer = tf.random_uniform_initializer()
        self.centroids = tf.Variable(centroid_initializer(shape=(self.item_code_bytes, 256, self.sub_embedding_size)), name="ItemCodes/centroids")

    def assign_codes(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j][1])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.data_parameters.num_items+1))
        print("fitting svd for initial centroids assignments")
        svd = TruncatedSVD(n_components=self.item_code_bytes)
        svd.fit(matr)
        item_embeddings = svd.components_
        assignments = []
        print("done")
        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
            ith_component = item_embeddings[i:i+1][0]
            ith_component = (ith_component - np.min(ith_component))/np.max(ith_component)
            noise = np.random.normal(0, 1e-5, self.data_parameters.num_items + 1)
            ith_component += noise # make sure that every item has unique value
            ith_component = np.expand_dims(ith_component, 1)
            component_assignments = discretizer.fit_transform(ith_component).astype('uint8')[:,0]
            assignments.append(component_assignments)
        centroid_asignments = np.transpose(np.array(assignments))
        self.item_codes.assign(centroid_asignments)

    def call(self, input_ids, batch_size):
        input_codes = tf.stop_gradient(tf.cast(tf.gather(self.item_codes, input_ids), 'int32'))
        code_byte_indices = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, self.item_code_bytes), 0), 0), [batch_size,self.data_parameters.sequence_length,1])
        n_sub_embeddings = batch_size * self.data_parameters.sequence_length * self.item_code_bytes
        code_byte_indices_reshaped = tf.reshape(code_byte_indices, (n_sub_embeddings, ))
        input_codes_reshaped = tf.reshape(input_codes, (n_sub_embeddings,))
        indices = tf.stack([code_byte_indices_reshaped, input_codes_reshaped], axis=-1)
        input_sub_embeddings_reshaped = tf.gather_nd(self.centroids, indices)
        result = tf.reshape(input_sub_embeddings_reshaped,[batch_size, self.data_parameters.sequence_length, self.item_code_bytes * self.sub_embedding_size] )
        return result

class SASLiftModel(SequentialRecsysModel):
    @classmethod
    def get_model_config_class(cls) -> Type[SASLiftConfig]:
        return SASLiftConfig

    def fit_biases(self, train_users):
        self.item_codes_layer.assign_codes(train_users)

    def __init__(self, model_parameters, data_parameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: SASLiftConfig #just a hint for the static analyser
        self.positions = tf.constant(tf.expand_dims(tf.range(self.data_parameters.sequence_length), 0))
        if self.model_parameters.pos_emb_comb != 'ignore':
            self.postion_embedding_layer = get_pos_embedding(self.data_parameters.sequence_length, self.model_parameters.embedding_size, self.model_parameters.pos_embedding)
        self.embedding_dropout = layers.Dropout(self.model_parameters.dropout_rate, name='embedding_dropout')
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
        if self.model_parameters.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.model_parameters.embedding_size, activation='gelu')
        self.sampler = get_negatives_sampler(self.model_parameters.vanilla_target_sampler, 
                                                 self.data_parameters, self.model_parameters.vanilla_num_negatives)
        self.item_codes_layer = ItemCodeLayer(self.model_parameters, self.data_parameters)


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

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb, attentions = self.get_seq_embedding(input_ids, bs=self.data_parameters.batch_size, training=training)
        seq_sub_emb = tf.reshape(seq_emb, [self.data_parameters.batch_size, self.data_parameters.sequence_length, self.item_codes_layer.item_code_bytes, self.item_codes_layer.sub_embedding_size])
        centroid_scores = tf.einsum("bsie,ine->bsin", seq_sub_emb, self.item_codes_layer.centroids) 
        target_positives = tf.expand_dims(inputs[1], -1)
        target_negatives = self.sampler(input_ids, target_positives)
        target_ids = tf.concat([target_positives, target_negatives], -1)
        target_codes =tf.transpose(tf.cast(tf.gather(self.item_codes_layer.item_codes, tf.nn.relu(target_ids)), 'int32'), [0, 1, 3, 2])
        target_sub_scores = tf.gather(centroid_scores, target_codes, batch_dims=3)
        logits = tf.reduce_sum(target_sub_scores, -2)
        positive_logits = logits[:, :, 0]
        negative_logits = logits[:,:,1:]
        minus_positive_logprobs = tf.math.softplus(-positive_logits)
        minus_negative_logprobs = tf.reduce_sum(tf.math.softplus(negative_logits) , axis=-1)
        minus_average_logprobs = (minus_positive_logprobs + minus_negative_logprobs) / (self.model_parameters.vanilla_num_negatives +1)
        mask = 1 - tf.cast(input_ids == self.data_parameters.num_items, 'float32')
        result = tf.reduce_sum(mask*minus_average_logprobs)/tf.reduce_sum(mask)
        return result

    

        
    def score_all_items(self, inputs):
        input_ids = inputs[0]
        seq_emb, attentions = self.get_seq_embedding(input_ids)
        seq_emb = seq_emb[:, -1, :]
        seq_sub_emb = tf.reshape(seq_emb, [seq_emb.shape[0], self.item_codes_layer.item_code_bytes, self.item_codes_layer.sub_embedding_size])
        centroid_scores = tf.einsum("bie,ine->bin", seq_sub_emb, self.item_codes_layer.centroids)
        centroid_scores = tf.transpose(tf.reshape(centroid_scores, [centroid_scores.shape[0], centroid_scores.shape[1] * centroid_scores.shape[2]]))
        target_codes = tf.cast(tf.transpose(self.item_codes_layer.item_codes[:-1]), 'int32')
        offsets = tf.expand_dims(tf.range(self.item_codes_layer.item_code_bytes) * 256, -1)
        target_codes += offsets
        result = tf.zeros((self.data_parameters.num_items, centroid_scores.shape[1]))
        for i in range (self.item_codes_layer.item_code_bytes):
            result += tf.gather(centroid_scores, target_codes[i])
        return tf.transpose(result)

    def get_seq_embedding(self, input_ids, bs=None, training=None):
        if bs is None:
            bs = input_ids.shape[0]
        seq = self.item_codes_layer(input_ids, bs)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.data_parameters.num_items), dtype=tf.float32), -1)
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
                pq_m = 4,
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
        self.pq_m = pq_m

    def as_dict(self):
        result = self.__dict__
        return result
    
    def get_model_architecture(self) -> Type[SequentialRecsysModel]:
        return SASLiftModel


    
