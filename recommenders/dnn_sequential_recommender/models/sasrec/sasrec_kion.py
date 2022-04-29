import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec_multihead_attention import multihead_attention


class KionChallengeSASRec(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, l2_emb=0.0, dropout_rate=0.2, num_blocks=3, num_heads=8,
                 user_features_attention_heads = 2,
                 ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.l2_emb = l2_emb
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.user_extra_features = True 
        self.user_features_attention_heads = user_features_attention_heads


    def get_model(self):
        model = KionSasrecModel(self.num_items, self.batch_size,
                               self.user_feature_max_val,
                               self.max_user_features,
                               self.output_layer_activation,
                               self.embedding_size,
                               self.max_history_length,
                               self.l2_emb,
                               self.dropout_rate, self.num_blocks, self.num_heads,
                               self.user_features_attention_heads, 
                               self.item_features
                                )
        return model
    


class KionSasrecModel(tensorflow.keras.Model):
    def __init__(self, num_items, batch_size, user_features_max_val,
                 max_user_features,output_layer_activation='linear', embedding_size=64,
                 max_history_length=64, l2_emb=0.0, dropout_rate=0.5, num_blocks=2, num_heads=1,
                 user_features_attention_heads = 2, item_features ={},
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
        self.item_features = tf.constant(self.build_item_features(item_features))
        self.item_features_embedding = layers.Embedding(self.max_feature_val + 1, self.embedding_size)
        user_layers = {}
        user_layers['embedding'] = layers.Embedding(user_features_max_val + 1, self.embedding_size, dtype='float32')
        user_layers['attention'] = layers.MultiHeadAttention(user_features_attention_heads,
                                                             key_dim=self.embedding_size)
        user_layers['dense'] = layers.Dense(self.embedding_size, activation='relu')
        user_layers['pooling'] = layers.MaxPool1D(max_user_features)
        user_layers['norm1'] = layers.LayerNormalization()
        user_layers['norm2'] = layers.LayerNormalization()
        user_layers['dropout'] = layers.Dropout(self.dropout_rate)

        self.max_user_features = max_user_features
        self.user_layers = user_layers
        user_layers['attention'] = layers.MultiHeadAttention(user_features_attention_heads,
                                                             key_dim=self.embedding_size)

        self.user_and_seq_attention = layers.MultiHeadAttention(user_features_attention_heads,
                                                             key_dim=self.embedding_size)
        self.user_and_seq_dense = layers.Dense(self.embedding_size, activation='gelu')

 


    def build_item_features(self, item_features):
        item_features_nums = []
        max_feature_val = 0
        for item in item_features:
            item_features_nums.append(len(item_features[item]))
            max_feature_val = max(max_feature_val, np.max(item_features[item]))
        num_features_model = int(np.percentile(item_features_nums, 90))
        features_matrix = []
        for i in range(self.num_items + 1):
            features = item_features.get(i, [])
            if len(features) > num_features_model:
                features = np.random.choice(features, num_features_model, replace=False)
            if len(features) < num_features_model:
                features = np.pad(features, [0, num_features_model - len(features)])
            features_matrix.append(features)
        result = np.array(features_matrix)
        self.num_features = num_features_model
        self.max_feature_val = max_feature_val
        return result

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



    def user_embeddings(self, user_features_input):
        user_features = self.user_layers['embedding'](user_features_input)
        shortrcut = user_features
        user_features = self.user_layers['attention'](user_features, user_features)
        user_features = self.user_layers['norm1'](user_features)
        user_features = shortrcut + user_features
        shortrcut = user_features
        user_features = self.user_layers['dense'](user_features)
        user_features = self.user_layers['norm2'](user_features)
        user_features = user_features + shortrcut
        user_features = self.user_layers['pooling'](user_features)
        user_features = self.user_layers['dropout'](user_features)
        return user_features

    def embed_items(self, item_ids):
        seq = self.item_embeddings_layer(item_ids)
        item_features = tf.gather(self.item_features, item_ids)
        item_features = self.item_embeddings_layer(item_features)
        item_features = tf.reduce_mean(item_features, axis=-2)
        seq += item_features
        return seq

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        user_features_input = inputs[1]

        seq = self.embed_items(input_ids)
        pos_embeddings = self.postion_embedding_layer(self.positions[:input_ids.shape[0]])
        seq += pos_embeddings
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, 0), dtype=tf.float32), -1)
        for i in range(self.num_blocks):
            seq = self.block(seq, mask, i)


        user_embedding = self.user_embeddings(user_features_input)


        seq_emb = self.seq_norm(seq)

        user_and_sequence = self.user_and_seq_attention(user_embedding, seq_emb) 
        user_and_sequence = tf.squeeze(user_and_sequence, 1)
        user_and_sequence = self.user_and_seq_dense(user_and_sequence)


        all_item_embeddings = self.embed_items(self.all_items)
        output = user_and_sequence @ tf.transpose(all_item_embeddings)
        output = self.output_activation(output)
        return output


