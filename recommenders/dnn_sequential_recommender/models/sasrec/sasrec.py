import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import activations

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
class SASRec(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, l2_emb=0.0, dropout_rate=0.2, num_blocks=3, num_heads=8):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.l2_emb = l2_emb
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads


    def get_model(self):
        model = OwnSasrecModel(self.num_items, self.batch_size, self.output_layer_activation,
                               self.embedding_size,
                               self.max_history_length,
                               self.l2_emb, self.dropout_rate, self.num_blocks, self.num_heads)
        return model



class OwnSasrecModel(tensorflow.keras.Model):
    def __init__(self, num_items, batch_size, output_layer_activation='linear', embedding_size=64,
                 max_history_length=64, l2_emb=0.0, dropout_rate=0.5, num_blocks=2, num_heads=1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_layer_activation = output_layer_activation
        self.embedding_size = embedding_size
        self.max_history_length = max_history_length
        self.l2_emb = l2_emb
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_items = num_items
        self.batch_size = batch_size
        self.positions = tf.constant(tf.tile(tf.expand_dims(tf.range(self.max_history_length), 0), [self.batch_size, 1]))

        self.item_embeddings_layer = layers.Embedding(self.num_items + 1, output_dim=self.embedding_size,
                                                      embeddings_regularizer=l2(self.l2_emb), dtype='float32')
        self.postion_embedding_layer = layers.Embedding(self.max_history_length,
                                                        self.embedding_size,
                                                        embeddings_regularizer=l2(self.l2_emb), dtype='float32')
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
                "dense": layers.Dense(self.embedding_size, activation='relu'),
                "dropout": layers.Dropout(self.dropout_rate)
            }
            self.attention_blocks.append(block_layers)
        self.output_activation = activations.get(self.output_layer_activation)
        self.seq_norm = layers.LayerNormalization()
        self.all_items = tf.range(0, self.num_items)

    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x = multihead_attention(queries, keys, self.num_heads, self.attention_blocks[i]["attention_layers"],
                                     causality=True) + queries
        residual = x
        x = self.attention_blocks[i]["second_norm"](x)
        x = self.attention_blocks[i]["dense"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x += residual
        x *= mask
        return x

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        seq = self.item_embeddings_layer(input_ids)
        pos_embeddings = self.postion_embedding_layer(self.positions)
        seq += pos_embeddings
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.num_items), dtype=tf.float32), -1)
        for i in range(self.num_blocks):
            seq = self.block(seq, mask, i)

        seq_emb = seq[:, -1, :]
        seq_emb = self.seq_norm(seq_emb)
        all_item_embeddings = self.item_embeddings_layer(self.all_items)
        output = seq_emb @ tf.transpose(all_item_embeddings)
        output = self.output_activation(output)
        return output


