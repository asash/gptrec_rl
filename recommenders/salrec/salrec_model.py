import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2


class SalrecModel(tf.keras.Model):
    def __init__(self, n_items, embedding_size,
                 max_user_feature_hashes, num_blocks, num_heads, num_bottlenecks,
                 output_layer_activation,
                 user_cat_features_space,
                 max_history_length,
                 num_user_cat_hashes,
                 bottleneck_size,
                 regularization,
                 positional,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.n_items = n_items
        self.items_embedding = layers.Embedding(self.n_items + 1, self.embedding_size)
        self.user_features_embedding = layers.Embedding(user_cat_features_space + 1, embedding_size)
        self.user_features_pooling = layers.AveragePooling1D(num_user_cat_hashes)
        self.max_user_feature_hashes = max_user_feature_hashes
        self.position_embedding_layer = layers.Embedding(2 * max_history_length + 1, embedding_size)
        self.position_feed_forward = layers.Dense(embedding_size, activation='swish')
        self.target_position_feed_forward = layers.Dense(embedding_size, activation='swish')
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.positional = positional
        self.blocks_layers = []
        for i in range(self.num_blocks):
            block_layers = {}
            block_layers["attention"] = layers.MultiHeadAttention(self.num_heads, key_dim=self.embedding_size)
            block_layers["dense"] = layers.Dense(self.embedding_size)
            block_layers["norm"] = layers.LayerNormalization()
            self.blocks_layers.append(block_layers)

        self.pos_attention = layers.MultiHeadAttention(self.num_heads, key_dim=self.embedding_size)
        self.norm = layers.LayerNormalization()
        self.flat = layers.Flatten()
        self.num_bottlenecks = num_bottlenecks
        self.bottlenecks = []
        for i in range(self.num_bottlenecks):
            self.bottlenecks.append(layers.Dense(bottleneck_size, activation='swish'))
        self.output_layer_activation = output_layer_activation
        self.output_layer = layers.Dense(n_items, name="output", activation=self.output_layer_activation,
                     bias_regularizer=l2(regularization),
                     kernel_regularizer=l2(regularization))



    def block(self, x, block_layers):
        shortcut = x
        attention = block_layers["attention"](x, x)
        attention = block_layers["dense"](attention)
        output = shortcut * attention
        output = block_layers["norm"](output)
        return output

    def __call__(self, inputs, **kwargs):
        items_input = inputs[0]
        x = self.items_embedding(items_input)
        input_idx = 1

        if self.max_user_feature_hashes > 0:
            features_input = inputs[input_idx]
            input_idx += 1
            user_features = self.user_features_embedding(features_input)
            user_features = self.user_features_pooling(user_features)

        if self.positional:
            pos_input = inputs[input_idx]
            input_idx += 1
            position_embedding = self.position_embedding_layer(pos_input)
            x = layers.Multiply()([x, position_embedding])


            target_pos_input = inputs[input_idx]
            input_idx += 1
            target_pos_embedding = self.position_embedding_layer(target_pos_input)
            target_pos_embedding = self.target_position_feed_forward(target_pos_embedding)

        if self.max_user_feature_hashes > 0:
            x = layers.concatenate([x, user_features], axis=1)

        for block_num in range(self.num_blocks):
            x = self.block(x, self.blocks_layers[block_num])

        if self.positional:
            x = self.pos_attention(target_pos_embedding, x)
            x = self.norm(x)

        x = self.flat(x)
        for i in range(self.num_bottlenecks):
            x = self.bottlenecks[i](x)
        output = self.output_layer(x)
        return output