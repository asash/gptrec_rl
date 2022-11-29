import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from aprec.losses.bce import BCELoss
from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
from aprec.losses.loss import ListWiseLoss

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from transformers import BertConfig, TFBertMainLayer

NUM_SPECIAL_ITEMS = 3 # +1 for mask item, +1 for padding, +1 for ignore_item
class FullBERT(SequentialRecsysModel):
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
                 loss = BCELoss(),
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
        self.loss = loss


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
        return FullBertModel(self.batch_size, self.output_layer_activation, bert_config, self.max_history_length, self.loss)


class FullBertModel(Model):
    def __init__(self, batch_size, outputput_layer_activation, bert_config, sequence_length, loss: ListWiseLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_items = bert_config.vocab_size - NUM_SPECIAL_ITEMS 
        self.output_layer_activation = tf.keras.activations.get(outputput_layer_activation)
        self.token_type_ids = tf.constant(tf.zeros(shape=(batch_size, bert_config.max_position_embeddings)))
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))
        self.bert = TFBertMainLayer(bert_config, add_pooling_layer=False)
        self.loss = loss
        self.loss.set_num_items(self.num_items)
        self.loss.set_batch_size(self.batch_size*self.sequence_length)

    def call(self, inputs, **kwargs):
        labels = inputs[1]
        positive_idx = tf.expand_dims(tf.nn.relu(labels), -1) #avoid boundary problems, negative values will be filteret later anyway
        sample_num = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, len(labels),dtype='int64'), -1), [1, self.sequence_length]), -1)
        sequence_pos = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, self.sequence_length, dtype='int64'), 0), [len(labels), 1]), -1)
        indices = tf.concat([sample_num, sequence_pos, positive_idx], -1)
        values = tf.ones([len(labels), self.sequence_length])
        use_mask = tf.tile(tf.expand_dims(tf.cast(labels!=-100,'float32'), -1),[1, 1, self.num_items])
        ground_truth = tf.scatter_nd(indices, values, [len(labels), self.sequence_length, self.num_items])
        ground_truth = use_mask*ground_truth + -100 * (1-use_mask)
        ground_truth = tf.reshape(ground_truth, (ground_truth.shape[0] * ground_truth.shape[1], ground_truth.shape[2]))
        ground_truth = tf.constant(ground_truth)

        masked_sequences = inputs[0]
        bert_output = self.bert(masked_sequences).last_hidden_state
        embeddings = self.bert.embeddings.weight[:-NUM_SPECIAL_ITEMS]
        logits = tf.einsum("bse, ne -> bsn", bert_output, embeddings)
        logits = tf.reshape(logits, (logits.shape[0]*logits.shape[1], logits.shape[2]))
        return self.loss.loss_per_list(ground_truth,logits)
                
   
    def score_all_items(self, inputs): 
        sequence = inputs[0] 
        sequence_embeddings  = self.bert(sequence, position_ids=self.position_ids_for_pred).last_hidden_state[:,-1]
        return tf.einsum("ne, be -> bn", self.bert.embeddings.weight[:self.num_items], sequence_embeddings)
