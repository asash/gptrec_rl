import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import activations

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from aprec.recommenders.metrics.ndcg import KerasNDCG
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
class SASRec(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, 
                 dropout_rate=0.2,
                 num_blocks=2,
                 num_heads=1,
                 pos_embedding = 'default', 
                 pos_emb_comb = 'add',
                 reuse_item_embeddings=True, #use same item embeddings for
                                             # sequence embedding and for the embedding matrix
                 encode_output_embeddings=False, #encode item embeddings with a dense layer
                                                          #may be useful if we reuse item embeddings
                 vanilla=False, #vanilla sasrec model uses shifted sequence prediction at the training time. 
                 sampled_targets=None,
                 ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.reuse_item_embeddings=reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.sampled_targets = sampled_targets
        self.vanilla = vanilla
        self.pos_embedding = pos_embedding
        self.pos_emb_comb = pos_emb_comb


    encode_embedding_with_dense_layer = False,

    def get_model(self):
        model = OwnSasrecModel(self.num_items, self.batch_size, self.output_layer_activation,
                               self.embedding_size,
                               self.max_history_length,
                               self.dropout_rate,
                               self.num_blocks,
                               self.num_heads,
                               self.reuse_item_embeddings,
                               self.encode_output_embeddings,
                               self.sampled_targets, 
                               vanilla=self.vanilla,
                               pos_embedding=self.pos_embedding, 
                               pos_emb_comb = self.pos_emb_comb
        )
        return model
class SinePositionEncoding(keras.layers.Layer):
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

class ExpPositionEncoding(keras.layers.Layer):
    def __init__(self, seq_len, emb_size, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.emb_size = emb_size
        pows_initalizer = tf.random_uniform_initializer(-3, 3)
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
    
    

class OwnSasrecModel(keras.Model):
    def __init__(self, num_items, batch_size, output_layer_activation='linear', embedding_size=64,
                 max_history_length=64, dropout_rate=0.5, num_blocks=2, num_heads=1,
                 reuse_item_embeddings=False,
                 encode_output_embeddings=False,
                 sampled_target=None,
                 pos_embedding = 'default', 
                 pos_emb_comb = 'add',
                 vanilla = False, #vanilla implementation; 
                                  #at the training time we calculate one positive and one negative per sequence element
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(not (vanilla and sampled_target), "only vanilla or sampled targetd strategy can be used at once")
        self.output_layer_activation = output_layer_activation
        self.embedding_size = embedding_size
        self.max_history_length = max_history_length
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_items = num_items
        self.batch_size = batch_size
        self.sampled_target = sampled_target
        self.reuse_item_embeddings=reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.vanilla = vanilla
        self.positions = tf.constant(tf.tile(tf.expand_dims(tf.range(self.max_history_length), 0), [self.batch_size, 1]))
        self.item_embeddings_layer = layers.Embedding(self.num_items + 2, output_dim=self.embedding_size, dtype='float32')
        self.postion_embedding_layer = get_pos_embedding(self.max_history_length, self.embedding_size, pos_embedding)
        self.embedding_dropout = layers.Dropout(self.dropout_rate)
        self.pos_embedding_comb = pos_emb_comb

        self.attention_blocks = []
        for i in range(self.num_blocks):
            block_layers = {
                "first_norm": layers.LayerNormalization(),
                "attention_layers": {
                    "query_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "key_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "val_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "dropout": layers.Dropout(self.dropout_rate),
                },
                "second_norm": layers.LayerNormalization(),
                "dense1": layers.Dense(self.embedding_size, activation='relu'),
                "dense2": layers.Dense(self.embedding_size),
                "dropout": layers.Dropout(self.dropout_rate)
            }
            self.attention_blocks.append(block_layers)
        self.output_activation = activations.get(self.output_layer_activation)
        self.seq_norm = layers.LayerNormalization()
        self.all_items = tf.range(0, self.num_items)
        if not self.reuse_item_embeddings:
            self.output_item_embeddings = layers.Embedding(self.num_items, self.embedding_size)

        if self.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.embedding_size, activation='gelu')


    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x, attentions = multihead_attention(queries, keys, self.num_heads, self.attention_blocks[i]["attention_layers"],
                                     causality=True)
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

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb, attentions = self.get_seq_embedding(input_ids, training)

        if self.vanilla or (self.sampled_target is not None):
            target_ids = inputs[1]
        else:
            target_ids = self.all_items
        target_embeddings = self.get_target_embeddings(target_ids)
        if self.vanilla:
            positive_embeddings = target_embeddings[:,:,0,:]
            negative_embeddings = target_embeddings[:,:,1,:]
            positive_results = tf.reduce_sum(seq_emb*positive_embeddings, axis=-1)
            negative_results = tf.reduce_sum(seq_emb*negative_embeddings, axis=-1)
            output = tf.stack([positive_results, negative_results], axis=-1)
        elif self.sampled_target:
            seq_emb = seq_emb[:, -1, :]
            output = tf.einsum("ij,ikj ->ik", seq_emb, target_embeddings)
        else:
            seq_emb = seq_emb[:, -1, :]
            output = seq_emb @ tf.transpose(target_embeddings)
        output = self.output_activation(output)
        return output
    
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
        if self.reuse_item_embeddings:
            target_embeddings = self.item_embeddings_layer(target_ids)
        else:
            target_embeddings = self.output_item_embeddings(target_ids)
        if self.encode_output_embeddings:
            target_embeddings = self.output_item_embeddings_encode(target_embeddings)
        return target_embeddings

    def get_seq_embedding(self, input_ids, training=None):
        seq = self.item_embeddings_layer(input_ids)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.num_items), dtype=tf.float32), -1)
        pos_embeddings = self.postion_embedding_layer(self.positions)[:input_ids.shape[0]]
        if self.pos_embedding_comb == 'add':
            seq += pos_embeddings
        elif self.pos_embedding_comb == 'mult':
            seq *= pos_embeddings
        seq = self.embedding_dropout(seq)
        seq *= mask
        attentions = []
        for i in range(self.num_blocks):
            seq, attention = self.block(seq, mask, i)
            attentions.append(attention)
        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions 