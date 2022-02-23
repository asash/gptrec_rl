from ast import List
from ctypes import Union
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from transformers.models.bert.modeling_tf_bert import BertConfig, TFBertMLMHead

from tensorflow.keras import Model
from tensorflow.keras import activations
from tensorflow.keras.layers import Embedding, Dense, Permute, Add, LayerNormalization, Activation, Layer
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


class RecsysMixer(SequentialRecsysModel):
    def __init__(self, output_layer_activation = 'linear',
                 embedding_size = 64, max_history_len = 100, token_mixer_hidden_dim = 64, channel_mixer_hidden_dim=64, num_blocks=2):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.embedding_size = embedding_size
        self.max_history_length = max_history_len
        self.output_layer_activation = output_layer_activation 
        self.token_mixer_hidden_dim = token_mixer_hidden_dim 
        self.channel_mixer_hidden_dim = channel_mixer_hidden_dim
        self.num_blocks = num_blocks

    def get_model(self):
        return MixerModel(self.embedding_size,
                          self.max_history_length, 
                          self.output_layer_activation, 
                          self.num_items, 
                          self.num_blocks, 
                          self.token_mixer_hidden_dim, 
                          self.channel_mixer_hidden_dim)

class MlpBlock(Layer):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation=None,
        **kwargs
    ):
        super(MlpBlock, self).__init__(**kwargs)

        if activation is None:
            activation = activations.gelu

        self.dim = dim
        self.hidden_dim = dim
        self.dense1 = Dense(hidden_dim)
        self.activation = Activation(activation)
        self.dense2 = Dense(dim)

    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_signature):
        return (input_signature[0], self.dim)

    def get_config(self):
        config = super(MlpBlock, self).get_config()
        config.update({
            'dim': self.dim,
            'hidden_dim': self.hidden_dim
        })
        return config


class MixerBlock(Layer):
    def __init__(
        self,
        num_patches: int,
        channel_dim: int,
        token_mixer_hidden_dim: int,
        channel_mixer_hidden_dim: int = None,
        activation=None,
        **kwargs
    ):
        super(MixerBlock, self).__init__(**kwargs)

        self.num_patches = num_patches
        self.channel_dim = channel_dim
        self.token_mixer_hidden_dim = token_mixer_hidden_dim
        self.channel_mixer_hidden_dim = channel_mixer_hidden_dim
        self.activation = activation

        if activation is None:
            self.activation = keras.activations.gelu

        if channel_mixer_hidden_dim is None:
            channel_mixer_hidden_dim = token_mixer_hidden_dim

        self.norm1 = LayerNormalization(axis=1)
        self.permute1 = Permute((2, 1))
        self.token_mixer = MlpBlock(num_patches, token_mixer_hidden_dim, name='token_mixer')

        self.permute2 = Permute((2, 1))
        self.norm2 = LayerNormalization(axis=1)
        self.channel_mixer = MlpBlock(channel_dim, channel_mixer_hidden_dim, name='channel_mixer')

        self.skip_connection1 = Add()
        self.skip_connection2 = Add()

    def call(self, inputs):
        x = inputs
        skip_x = x
        x = self.norm1(x)
        x = self.permute1(x)
        x = self.token_mixer(x)

        x = self.permute2(x)

        x = self.skip_connection1([x, skip_x])
        skip_x = x

        x = self.norm2(x)
        x = self.channel_mixer(x)

        x = self.skip_connection2([x, skip_x])  # TODO need 2?

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MixerBlock, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'channel_dim': self.channel_dim,
            'token_mixer_hidden_dim': self.token_mixer_hidden_dim,
            'channel_mixer_hidden_dim': self.channel_mixer_hidden_dim,
            'activation': self.activation,
        })
        return config



class MixerModel(Model):
    def __init__(self, embedding_size, max_history_length, output_layer_activation, num_items, num_blocks,
                        token_mixer_hidden_dim, 
                        channel_mixer_hidden_dim, 
                        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positions_embedding = Embedding(2 * max_history_length, embedding_size) 
        self.item_embeddings = Embedding(num_items + 2, embedding_size) 
        self.mixer_blocks = []
        for _ in range(num_blocks):
            self.mixer_blocks.append(MixerBlock(max_history_length, embedding_size, token_mixer_hidden_dim, channel_mixer_hidden_dim))
        self.num_blocks = num_blocks
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, max_history_length +1))).reshape(1, max_history_length))
        self.output_layer = Dense(num_items)

        

    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        labels = inputs[1]        
        positions = inputs[2]
        result = self.get_scores(sequences, positions)
        loss = hf_compute_loss(labels, result)
        return loss

    def get_scores(self, sequences, positions):
        x = self.positions_embedding(positions) +  self.item_embeddings(sequences)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x) 
        predictions = self.output_layer(x) 
        return predictions


    def score_all_items(self, inputs):
        sequence = inputs[0] 
        result = self.get_scores(sequence, self.position_ids_for_pred)[:,-1,:-2]
        return result

def hf_compute_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    # make sure only labels that are not equal to -100 affect the loss
    active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
    return loss_fn(labels, reduced_logits)



def shape_list(tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
