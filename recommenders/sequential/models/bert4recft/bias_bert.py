import numpy as np
import tensorflow as tf
from aprec.losses.bce import BCELoss
from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.loss import ListWiseLoss
from aprec.losses.softmax_crossentropy import SoftmaxCrossEntropy
from aprec.recommenders.sequential.models.bert4recft.special_items import SPECIAL_ITEMS

from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialRecsysModelBuilder
from transformers import BertConfig, TFBertMainLayer
from scipy.sparse import lil_matrix, csr_matrix

NUM_SPECIAL_ITEMS = len(SPECIAL_ITEMS) 


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

class BiasBERT(SequentialRecsysModelBuilder):
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
                 loss = SoftmaxCrossEntropy()
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
        return BiasBERTModel(self.batch_size, self.output_layer_activation, bert_config, self.max_history_length, self.loss)


class BiasBERTModel(tf.keras.Model):
    def __init__(self, batch_size, outputput_layer_activation, bert_config, sequence_length,  loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_items = bert_config.vocab_size - NUM_SPECIAL_ITEMS 
        self.output_layer_activation = tf.keras.activations.get(outputput_layer_activation)
        self.token_type_ids = tf.constant(tf.zeros(shape=(batch_size, bert_config.max_position_embeddings)))
        #self.bert = TFBertMainLayer(bert_config, add_pooling_layer=False)
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))
        self.bias_agg_hidden = tf.keras.layers.Dense(32, 'gelu')
        self.bias_agg_output = tf.keras.layers.Dense(1)
        self.loss=loss

    def fit_biases(self, train_users):
        print("fitting one step transition biases...")
        self.pad = self.num_items + SPECIAL_ITEMS['PAD']
        transitions = lil_matrix((self.num_items+len(SPECIAL_ITEMS), self.num_items+len(SPECIAL_ITEMS)))
        for i in range(len(train_users)):
            seq = [(0, self.pad)] + train_users[i] + [(0, self.pad)]
            for src in range(len(seq)-1):
                dst = src + 1
                transitions[seq[src][1],seq[dst][1]] += 1
            pass
        smoothed_transitions, pop_bias = self.get_smoothed_transitions(transitions)
        self.smoothed_transitions_src = tf.constant(smoothed_transitions.todense())
        self.smoothed_transitions_dst = tf.constant(np.transpose(smoothed_transitions).todense())
        self.pop_biases = pop_bias.T
        self.pop_biases_norm = tf.expand_dims(tf.constant(pop_bias.T/np.sum(pop_bias), 'float32'), 0)

    def get_smoothed_transitions(self, transitions):
        nz_rows, nz_cols = transitions.nonzero()
        pop_bias = np.sum(transitions, -1)
        smoothing_data = csr_matrix((np.ones_like(nz_cols), (nz_rows, nz_cols)), shape=transitions.shape)
        smoothed_transitions = transitions - smoothing_data
        return smoothed_transitions, pop_bias

    def call(self, inputs, **kwargs):
        masked_sequences = inputs[0]
        labels = inputs[1]
        positions = inputs[2]
        positive_idx = tf.expand_dims(tf.nn.relu(labels), -1) #avoid boundary problems, negative values will be filteret later anyway
        sample_num = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, len(labels),dtype='int64'), -1), [1, self.sequence_length]), -1)
        sequence_pos = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, self.sequence_length, dtype='int64'), 0), [len(labels), 1]), -1)
        indices = tf.concat([sample_num, sequence_pos, positive_idx], -1)
        values = tf.ones([len(labels), self.sequence_length])
        use_mask = tf.tile(tf.expand_dims(tf.cast(labels!=-100,'float32'), -1),[1, 1, self.num_items])
        ground_truth = tf.scatter_nd(indices, values, [len(labels), self.sequence_length, self.num_items])
        ground_truth = use_mask*ground_truth + -100 * (1-use_mask)

        biases = self.get_biases(masked_sequences)
        #bert_output = self.bert(masked_sequences, position_ids = positions).last_hidden_state
        #embeddings = self.bert.embeddings.weight[:-NUM_SPECIAL_ITEMS]
        return self.get_loss(ground_truth,biases)

    def get_biases(self, masked_sequences):
        seqs = tf.nn.relu(masked_sequences)
        pad = tf.cast(tf.fill((masked_sequences.shape[0], 1), self.pad), 'int64')
        shift_src = tf.concat([pad, seqs[:, :-1]], 1)
        shift_dst = tf.concat([seqs[:, 1:], pad], 1)
        mc_src_probs = tf.cast(tf.math.divide_no_nan(tf.gather(self.smoothed_transitions_src, shift_src), self.pop_biases), 'float32')
        mc_dst_probs = tf.cast(tf.math.divide_no_nan(tf.gather(self.smoothed_transitions_dst, shift_dst), self.pop_biases), 'float32')
        pop_bias = tf.tile(self.pop_biases_norm, [mc_src_probs.shape[0], mc_src_probs.shape[1], 1])
        concat = tf.stack([mc_src_probs, mc_dst_probs, pop_bias], -1)
        bias_hidden = self.bias_agg_hidden(concat)
        result = tf.squeeze(self.bias_agg_output(bias_hidden), -1)
        return result[:, :,:-len(SPECIAL_ITEMS)]


    def get_loss(self, ground_truth, logits):
        ground_truth = tf.reshape(ground_truth, (ground_truth.shape[0] * ground_truth.shape[1], ground_truth.shape[2]))
        logits = tf.reshape(logits, (logits.shape[0]*logits.shape[1], logits.shape[2]))
        result = self.loss.loss_per_list(ground_truth, logits)
        return result 
 
    def score_all_items(self, inputs): 
        sequence = inputs[0] 
        biases = self.get_biases(sequence)
        return biases[:,-1]
        #sequence_embeddings  = self.bert(sequence, position_ids=self.position_ids_for_pred).last_hidden_state[:,-1]
        #return tf.einsum("ne, be -> bn", self.bert.embeddings.weight[:self.num_items], sequence_embeddings)
    
    def log(self):
        pass
        #tf.summary.scalar('biases/popularity_weight', self.pop_bias_weight)
        #tf.summary.scalar('biases/src_weight', self.src_bias_weight)
        #tf.summary.scalar('biases/dst_weight', self.dst_bias_weight)

