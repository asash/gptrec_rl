import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import activations

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
class SASRec(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, l2_emb=0.0, dropout_rate=0.5, num_blocks=2, num_heads=1):
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
        self.all_items = tf.range(1, self.num_items+1)

    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x = self.multihead_attention(queries, keys, self.attention_blocks[i]["attention_layers"],
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
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, 0), dtype=tf.float32), -1)
        for i in range(self.num_blocks):
            seq = self.block(seq, mask, i)

        seq_emb = seq[:, -1, :]
        seq_emb = self.seq_norm(seq_emb)
        all_item_embeddings = self.item_embeddings_layer(self.all_items)
        output = seq_emb @ tf.transpose(all_item_embeddings)
        output = self.output_activation(output)
        return output

    #this version of multihead attention was ported from the original SASRec implementation,
    # as it does some non-standard transformations, including 'casuality' one
    def multihead_attention(self,
                            queries,
                            keys,
                            attention_layers,
                            causality=False,
                            ):

        Q = attention_layers["query_proj"](queries) # (N, T_q, C)
        K = attention_layers["key_proj"](keys)  # (N, T_k, C)
        V = attention_layers["val_proj"](keys) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = attention_layers["dropout"](outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, C)
        return outputs

