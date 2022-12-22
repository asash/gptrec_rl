from __future__ import annotations
from typing import Type
import tensorflow as tf
from aprec.losses import get_loss
from aprec.recommenders.sequential.samplers.sampler import get_negatives_sampler
layers = tf.keras.layers


from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialModelConfig, SequentialRecsysModel
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec

class SASRecModel(SequentialRecsysModel):

    @classmethod
    def get_model_config_class(cls) -> Type[SASRecConfig]:
        return SASRecConfig
     
    def __init__(self, model_parameters, data_parameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: SASRecConfig #just a hint for the static analyser
        self.positions = tf.constant(tf.expand_dims(tf.range(self.data_parameters.sequence_length), 0))
        self.item_embeddings_layer = layers.Embedding(
                                                      self.data_parameters.num_items + 2, 
                                                      output_dim=self.model_parameters.embedding_size, dtype='float32', 
                                                      name='item_embeddings')
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
        if not self.model_parameters.reuse_item_embeddings:
            self.output_item_embeddings = layers.Embedding(self.data_parameters.num_items + 2, self.model_parameters.embedding_size)

        if self.model_parameters.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.model_parameters.embedding_size, activation='gelu')
        self.loss_ = get_loss.listwise_loss_from_config(self.model_parameters.loss, self.model_parameters.loss_params)
        if self.model_parameters.vanilla:
            self.sampler = get_negatives_sampler(self.model_parameters.vanilla_target_sampler, 
                                                 self.data_parameters, self.model_parameters.vanilla_num_negatives)
            self.loss_.set_batch_size(self.data_parameters.batch_size * self.data_parameters.sequence_length)
            self.loss_.set_num_items(self.model_parameters.vanilla_num_negatives + 1)
        else:
            self.loss_.set_batch_size(self.data_parameters.batch_size)
            self.loss_.set_num_items(self.data_parameters.num_items)
        pass

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
        if self.model_parameters.vanilla:
            positives = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int64')
            inputs.append(positives)
        else:
            pad = tf.cast(tf.fill((self.data_parameters.batch_size, self.model_parameters.max_targets_per_user - 1), self.data_parameters.num_items), 'int64')
            user_positives = tf.zeros((self.data_parameters.batch_size, 1), 'int64')
            padded_positives = tf.concat([user_positives, pad], - 1)
            inputs.append(padded_positives)
        return inputs

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb, attentions = self.get_seq_embedding(input_ids, bs=self.data_parameters.batch_size, training=training)
        if self.model_parameters.vanilla:
            target_positives = tf.expand_dims(inputs[1], -1)
            target_negatives = self.sampler(input_ids, target_positives)
            target_ids = tf.concat([target_positives, target_negatives], -1)
            pass
        else:
            target_ids = self.all_items
            positive_input_ids = inputs[1]
        
        #use relu to hide negative targets - they will be ignored later in any case
        target_embeddings = self.get_target_embeddings(tf.nn.relu(target_ids))
        if self.model_parameters.vanilla:
            cnt_per_pos = self.model_parameters.vanilla_num_negatives + 1
            output = tf.einsum("bse, bsne -> bsn", seq_emb, target_embeddings)
            truth_positives = tf.ones((self.data_parameters.batch_size, self.data_parameters.sequence_length, 1))
            truth_negatives = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length, self.model_parameters.vanilla_num_negatives))
            ground_truth = tf.concat([truth_positives, truth_negatives], axis=-1)
            mask = tf.expand_dims(tf.cast((input_ids == self.data_parameters.num_items), 'float32'), -1)
            mask = tf.tile(mask, [1, 1, cnt_per_pos])
            ground_truth = -100 * mask + ground_truth * (1-mask) #ignore padding in loss
            ground_truth = tf.reshape(ground_truth, (self.data_parameters.sequence_length * self.data_parameters.batch_size, cnt_per_pos))
            output = tf.reshape(output, (self.data_parameters.sequence_length * self.data_parameters.batch_size, cnt_per_pos))
            pass

        else:
            seq_emb = seq_emb[:, -1, :]
            output = seq_emb @ tf.transpose(target_embeddings)
            batch_idx = tf.tile(tf.expand_dims(tf.range(self.data_parameters.batch_size), -1), [1, self.model_parameters.max_targets_per_user])
            batch_idx = tf.reshape(batch_idx, (self.data_parameters.batch_size * self.model_parameters.max_targets_per_user,))
            item_idx = tf.cast(tf.reshape(positive_input_ids,
                                          (self.data_parameters.batch_size * self.model_parameters.max_targets_per_user,)), 'int32')
            idx = tf.stack((batch_idx, item_idx), -1)
            vals = tf.ones(self.data_parameters.batch_size * self.model_parameters.max_targets_per_user , 'float32')
            ground_truth = tf.scatter_nd(idx , vals, (self.data_parameters.batch_size, self.data_parameters.num_items+1))[:,:-1]
            
        output = self.output_activation(output)
        result =  self.loss_.loss_per_list(ground_truth, output)
        return result
    
    def score_all_items(self, inputs):
        input_ids = inputs[0]
        seq_emb, attentions = self.get_seq_embedding(input_ids)
        seq_emb = seq_emb[:, -1, :]
        target_ids = self.all_items
        target_embeddings = self.get_target_embeddings(target_ids)
        output = seq_emb @ tf.transpose(target_embeddings)
        output = self.output_activation(output)
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
        before = seq
        seq = self.embedding_dropout(seq)
        seq *= mask
        attentions = []
        for i in range(self.model_parameters.num_blocks):
            seq, attention = self.block(seq, mask, i)
            attentions.append(attention)
        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions 

class SASRecConfig(SequentialModelConfig):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                dropout_rate=0.5, num_blocks=2, num_heads=1,
                reuse_item_embeddings=False,
                encode_output_embeddings=False,
                pos_embedding = 'default', 
                pos_emb_comb = 'add',
                causal_attention = True,
                pos_smoothing = 0,
                max_targets_per_user=10,
                vanilla = False,
                vanilla_num_negatives = 1,
                vanilla_target_sampler = 'random',
                loss='bce', 
                loss_params = {}
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
        self.vanilla = vanilla
        self.max_targets_per_user = max_targets_per_user #only used with sparse positives
        self.loss = loss
        self.loss_params = loss_params
        self.vanilla_num_negatives = vanilla_num_negatives 
        self.vanilla_target_sampler = vanilla_target_sampler

    def as_dict(self):
        result = self.__dict__
        return result
    
    def get_model_architecture(self) -> Type[SequentialRecsysModel]:
        return SASRecModel 


    
class SinePositionEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        seq_length, 
        hidden_size,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.seq_length = seq_length
        self.hidden_size = hidden_size
    
    def call(self, positions):
        seq_length = self.seq_length
        hidden_size = self.hidden_size
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
            / tf.cast(hidden_size, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )
        return tf.gather(positional_encodings, positions)

class ExpPositionEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, emb_size, init=3, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.emb_size = emb_size
        pows_initalizer = tf.random_uniform_initializer(-init, init)
        self.pow = tf.Variable(initial_value=pows_initalizer(shape=(emb_size, )), trainable=True)
        
    
    def __call__(self, positions):
        w = tf.exp(self.pow)
        for i in range(len(positions.shape)):
            w = tf.expand_dims(w, 0)
        tiles = list(positions.shape) + [1]
        w = tf.tile(w, tiles)
        positions_norm = tf.cast((positions+1), 'float32')/(self.seq_len+1)
        pos = tf.tile(tf.expand_dims(positions_norm, -1), [1] * len(positions.shape) + [self.emb_size])
        return tf.pow(pos, w)
    
def get_pos_embedding(seq_len, emb_size, kind):
    if kind == 'default':
        return layers.Embedding(seq_len, output_dim=emb_size, dtype='float32')

    if kind == 'exp':
        return ExpPositionEncoding(seq_len, emb_size)
    
    if kind == 'sin':
        return SinePositionEncoding(seq_len, emb_size)

        
