import numpy as np
from tensorflow.keras import Model
import tensorflow as tf

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from transformers import BertConfig, TFBertMainLayer

class BERT4RecFT(SequentialRecsysModel):
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


    def get_model(self):
        bert_config = BertConfig(
            vocab_size = self.num_items + 3, # +1 for mask item, +1 for padding, +1 for ignore_item
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
        return BERT4RecFTModel(self.batch_size, self.output_layer_activation, bert_config, self.max_history_length)


class BERT4RecFTModel(Model):
    def __init__(self, batch_size, outputput_layer_activation, bert_config, sequence_length, 
                        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.output_layer_activation = tf.keras.activations.get(outputput_layer_activation)
        self.token_type_ids = tf.constant(tf.zeros(shape=(batch_size, bert_config.max_position_embeddings)))
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))
        self.bert = TFBertMainLayer(bert_config, add_pooling_layer=False, name="bert")

    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        candidates = inputs[1]
        positions = inputs[2]
        bert_output = self.bert(sequences, position_ids=positions)              
        emb_matrix = tf.gather(self.bert.embeddings.weight, candidates)
        result = tf.einsum("ijk,ijmk->ijm", bert_output[0], emb_matrix)
        result = self.output_layer_activation(result)
        return result
    
    def get_embedding_matrix(self):
        return self.bert.embeddings.weight[:-3,:]
    
    def get_sequence_embeddings(self, inputs):
        sequence = inputs[0] 
        bert_output = self.bert(sequence, position_ids=self.position_ids_for_pred)[0][:,-1,:]             
        return bert_output
 

    def score_all_items(self, inputs): 
        embedding_matrix = self.get_embedding_matrix()
        sequence_embeddings = self.get_sequence_embeddings(inputs)
        result = tf.einsum("ij,kj->ki", embedding_matrix, sequence_embeddings)
        return result