import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from aprec.losses.bce import BCELoss
from aprec.losses.softmax_crossentropy import SoftmaxCrossEntropy
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss 

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from transformers import BertConfig, TFBertMainLayer

NUM_SPECIAL_ITEMS = 3 # +1 for mask item, +1 for padding, +1 for ignore_item
class TwoBERTS(SequentialRecsysModel):
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
                 num_samples=256, 
                 retriever_loss = SoftmaxCrossEntropy(),
                 reranker_loss = LambdaGammaRankLoss() 
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
        self.num_samples = num_samples
        self.retriever_loss = retriever_loss
        self.reranker_loss = reranker_loss


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
        return TwoBERTsModel(self.batch_size, self.num_samples,
                             self.output_layer_activation,
                             bert_config, self.max_history_length, 
                             self.retriever_loss,
                             self.reranker_loss 
                             )


class TwoBERTsModel(Model):
    def __init__(self, batch_size, num_samples, outputput_layer_activation, bert_config, sequence_length, 
                        retriever_loss, 
                        reranker_loss,
                        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_items = bert_config.vocab_size - NUM_SPECIAL_ITEMS 
        self.output_layer_activation = tf.keras.activations.get(outputput_layer_activation)
        self.token_type_ids = tf.constant(tf.zeros(shape=(batch_size, bert_config.max_position_embeddings)))
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))
        self.bert_retriever = TFBertMainLayer(bert_config, add_pooling_layer=False, name="retriever")
        self.bert_reranker = TFBertMainLayer(bert_config, add_pooling_layer=False, name="reranker")
        self.retriever_loss = retriever_loss()
        self.reranker_loss = reranker_loss()
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, sequence_length +1))).reshape(1, sequence_length))

    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        positions = inputs[2]
        labels = inputs[1]
        positive_candidates = tf.expand_dims(labels, -1)
        positives_ground_truth = tf.ones_like(positive_candidates, dtype='float32')
        candidates_shape = (self.batch_size, self.sequence_length, self.num_samples)
        negative_candidates = tf.random.uniform(candidates_shape, 0, self.num_items, dtype='int64')
        negatives_ground_truth = tf.zeros_like(negative_candidates, dtype='float32')
        candidates = tf.concat([positive_candidates, negative_candidates], -1)
        retriever_bert_output = self.bert_retriever(sequences, position_ids=positions).last_hidden_state              
        emb_matrix = tf.gather(self.bert_retriever.embeddings.weight, candidates)
        retriever_result = tf.einsum("ijk,ijmk->ijm", retriever_bert_output, emb_matrix)
        retriever_result = tf.transpose(retriever_result, [0, 2, 1])
        ground_truth = tf.concat([positives_ground_truth, negatives_ground_truth], -1)
        ground_truth = tf.transpose(ground_truth, [0, 2, 1])
        retriever_losses = self.retriever_loss.calc_per_list(ground_truth, retriever_result)
        loss_mask = tf.cast(labels != -100, 'float32')
        retriever_losses_masked = retriever_losses*loss_mask
        retriever_loss = tf.math.divide_no_nan(tf.reduce_sum(retriever_losses_masked), tf.reduce_sum(loss_mask))
        all_retriever_embeddings = self.bert_retriever.embeddings.weight[:-NUM_SPECIAL_ITEMS]
        all_items_retriever_scores = tf.einsum("ijk,nk->ijn", retriever_bert_output, all_retriever_embeddings)
        retrieved_candidates = tf.cast(tf.math.top_k(all_items_retriever_scores, self.num_samples, sorted=False).indices, 'int64')
        tiled_positives = tf.tile(positive_candidates, [1, 1, self.num_samples])
        reranker_ground_truth = tf.cast(tiled_positives == retrieved_candidates, 'float32')
        reranker_ground_truth = tf.transpose(reranker_ground_truth, [0, 2, 1])
        
        reranker_bert_output = self.bert_reranker(sequences, position_ids=positions).last_hidden_state              
        reranker_candidate_embs = tf.gather(self.bert_reranker.embeddings.weight, retrieved_candidates)
        reranker_result = tf.einsum("ijk,ijmk->ijm", reranker_bert_output, reranker_candidate_embs) 
        reranker_result = tf.transpose(reranker_result, [0, 2, 1])
        reranker_losses = self.reranker_loss.calc_per_list(reranker_ground_truth, reranker_result)
        reranker_losses_masked = reranker_losses*loss_mask
        reranker_loss = tf.math.divide_no_nan(tf.reduce_sum(reranker_losses_masked), tf.reduce_sum(loss_mask))
        return (retriever_loss + reranker_loss) / 2 
    
    def score_all_items(self, inputs): 
        sequence = inputs[0] 
        retriever_sequence_embedding = self.bert_retriever(sequence, position_ids=self.position_ids_for_pred).last_hidden_state[:,-1]
        retriever_scores = tf.einsum("ne, be -> bn", self.bert_retriever.embeddings.weight[:self.num_items], retriever_sequence_embedding)

        candidates = tf.math.top_k(retriever_scores, self.num_samples, sorted=True).indices
        reranker_candidate_embs = tf.gather(self.bert_reranker.embeddings.weight, candidates)

        reranker_sequence_embedding = self.bert_reranker(sequence, position_ids=self.position_ids_for_pred).last_hidden_state[:,-1]
        reranker_scores = tf.sigmoid(tf.einsum("bce, be -> bc", reranker_candidate_embs, reranker_sequence_embedding))
        batch_size= sequence.shape[0] 
        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, self.num_samples]),
                                        (self.num_samples * batch_size, 1))

        reranker_indices = tf.reshape(candidates, (self.num_samples * batch_size, 1))
        indices = tf.concat([batch_indices, reranker_indices], axis=1)
        result = tf.scatter_nd(indices, tf.reshape(reranker_scores, [self.num_samples * batch_size]),
                                                                           tf.constant([batch_size, self.num_items]))

        return result