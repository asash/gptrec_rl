import numpy as np
import tensorflow as tf
from aprec.losses.bce import BCELoss
from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
from aprec.losses.loss import ListWiseLoss
import tensorflow_probability as tfp
from collections import Counter


from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from transformers import BertConfig, TFBertMainLayer

NUM_SPECIAL_ITEMS = 3 # +1 for mask item, +1 for padding, +1 for ignore_item
class QuantumBERT(SequentialRecsysModel):
    def __init__(self, output_layer_activation = 'linear',
                 embedding_size = 64, max_history_len = 100,
                 attention_probs_dropout_prob = 0.2,
                 hidden_act = "gelu",
                 hidden_dropout_prob = 0.2,
                 initializer_range = 0.02,
                 intermediate_size = 128,
                 num_attention_heads = 2,
                 num_hidden_layers = 3,
                 type_vocab_size = 2,
                 quants_per_dim = 10
                 ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.embedding_size = embedding_size
        self.max_history_length = max_history_len
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads 
        self.num_hidden_layers = num_hidden_layers 
        self.type_vocab_size = type_vocab_size      
        self.quants_per_dim = quants_per_dim


    def get_model(self):
        bert_config = BertConfig(
            vocab_size = self.num_items + NUM_SPECIAL_ITEMS, 
            hidden_size = self.embedding_size,
            max_position_embeddings=2*self.max_history_length, 
            attention_probs_dropout_prob=self.attention_probs_dropout_prob, 
            hidden_act=self.hidden_act, 
            hidden_dropout_prob=self.hidden_dropout_prob, 
            initializer_range=self.initializer_range, 
            num_attention_heads=self.num_attention_heads, 
            num_hidden_layers=self.num_hidden_layers, 
            type_vocab_size=self.type_vocab_size, 
        )
        return QuantumBERTModel(self.batch_size, self.output_layer_activation, bert_config, self.max_history_length, self.quants_per_dim)


class QuantumBERTModel(tf.keras.Model):
    def __init__(self, batch_size, outputput_layer_activation, bert_config, sequence_length, quants_per_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_items = bert_config.vocab_size - NUM_SPECIAL_ITEMS 
        self.output_layer_activation = tf.keras.activations.get(outputput_layer_activation)
        self.token_type_ids = tf.constant(tf.zeros(shape=(batch_size, bert_config.max_position_embeddings)))
        self.bert = TFBertMainLayer(bert_config, add_pooling_layer=False)
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))
        self.quants_per_dim = quants_per_dim
        self.percentiles = tf.linspace(0, 100, quants_per_dim+1)[1:]
        self.output_layer = tf.keras.layers.Dense(self.quants_per_dim * bert_config.hidden_size)
        self.embedding_size = bert_config.hidden_size

    def call(self, inputs, **kwargs):
        labels = inputs[1]
        masked_sequences = inputs[0]
        positions = inputs[2]
        attribute_probs = self.predict_attributes(masked_sequences, positions)
        attributes = self.quantize(labels)
        truth = self.attributes_one_hot(labels, attributes)
        use_loss = tf.expand_dims(tf.cast(labels != -100, 'float32'), -1)
        loss = tf.losses.binary_crossentropy(truth, attribute_probs)*use_loss
        result = tf.math.divide_no_nan(tf.reduce_sum(loss), tf.reduce_sum(use_loss)*self.embedding_size)
        return result

    def attributes_one_hot(self, labels, attributes, seq=True):
        idx_batch_dim = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, labels.shape[0], dtype='int32'), -1), -1), [1, labels.shape[1], self.embedding_size])
        idx_seq_dim = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, labels.shape[1], dtype='int32'), 0), -1), [labels.shape[0], 1, self.embedding_size]) 
        idx_emb_dim = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, self.embedding_size, dtype='int32'), 0), 0), [labels.shape[0], labels.shape[1], 1]) 
        index = tf.stack([idx_batch_dim, idx_seq_dim, idx_emb_dim, attributes], -1)
        values = tf.ones_like(attributes)
        truth = tf.scatter_nd(index, values, attributes.shape + [self.quants_per_dim])
        return truth

    def quantize(self, labels):
        embeddings_matrix = self.bert.embeddings.weight
        percentile_borders = tf.transpose(tfp.stats.percentile(embeddings_matrix, self.percentiles, axis=0))
        label_embeddings = tf.transpose(tf.gather(embeddings_matrix, labels), [2, 0, 1])
        label_embeddings = tf.reshape(label_embeddings, [label_embeddings.shape[0], labels.shape[0]*labels.shape[1]])
        quantized_label_embeddings = tf.searchsorted(percentile_borders, label_embeddings)
        quantized_label_embeddings = tf.clip_by_value(quantized_label_embeddings, 0, self.quants_per_dim - 1)
        quantized_label_embeddings = tf.reshape(quantized_label_embeddings, [quantized_label_embeddings.shape[0], labels.shape[0], labels.shape[1]])
        return tf.transpose(quantized_label_embeddings, [1, 2, 0])


    def predict_attributes(self, sequences, positions):
        bert_output = self.bert(sequences, position_ids = positions).last_hidden_state
        output_projection = self.output_layer(bert_output)
        output_projection = tf.reshape(output_projection, [output_projection.shape[0], output_projection.shape[1], self.embedding_size, self.quants_per_dim])
        output_projection = tf.math.softmax(output_projection)
        return output_projection


    def score_all_items(self, inputs): 
        sequences = inputs[0]
        attribute_probs = self.predict_attributes(sequences, self.position_ids_for_pred)[:, -1]
        all_items = tf.expand_dims(tf.range(0, self.num_items),0)
        all_item_attributes = self.quantize(all_items)
        all_items_attr_onehot =  tf.squeeze(self.attributes_one_hot(all_items, all_item_attributes), 0)
        result = tf.einsum("bea, iea -> bi", attribute_probs, tf.cast(all_items_attr_onehot, 'float32'))
        return result
