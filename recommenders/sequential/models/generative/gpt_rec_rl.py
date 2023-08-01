from __future__ import annotations

from typing import List, Type
import tensorflow as tf
from aprec.recommenders.sequential.models.generative.tokenizer import Tokenizer
from aprec.recommenders.sequential.models.generative.tokenizers.tokenizer_utils import get_tokenizer_class
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
from transformers import GPT2Config, TFGPT2LMHeadModel 

class RLGPT2RecConfig(SequentialModelConfig):
    def __init__(self,
                 embedding_size = 256, 
                 transformer_blocks = 3, 
                 attention_heads = 4, 
                 tokenizer='svd',
                 values_per_dim = 1024, 
                 tokens_per_item = 4, 
                 generate_top_k = 50, 
                 generate_top_p = 0.95,
                 generate_n_sequences = 50,
                 generation_temperature=1.0, 
                 generate_max_tokens = 10
                 ):
        self.embedding_size = embedding_size
        self.transformer_blocks = transformer_blocks
        self.attention_heads = attention_heads
        self.values_per_dim = values_per_dim
        self.tokens_per_item = tokens_per_item
        self.tokenizer = tokenizer
        self.generate_top_k = generate_top_k
        self.generate_top_p = generate_top_p
        self.generate_n_sequences = generate_n_sequences 
        self.generation_temperature = generation_temperature
        self.generate_max_tokens = generate_max_tokens
        
    def as_dict(self):
        return self.__dict__
    
    def get_model_architecture(self) -> Type[RLGPT2RecConfig]:
        return RLGPT2RecModel

class RLGPT2RecModel(SequentialRecsysModel):
    def __init__(self, model_parameters: RLGPT2RecConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: RLGPT2RecConfig
        self.tokenizer_class = get_tokenizer_class(model_parameters.tokenizer)
        self.tokenizer:Tokenizer = self.tokenizer_class(model_parameters.tokens_per_item, model_parameters.values_per_dim, data_parameters.num_items)
        gpt_config = GPT2Config(
            vocab_size = int(self.tokenizer.vocab_size) + 1, #+1 for padding
            n_positions = data_parameters.sequence_length * model_parameters.tokens_per_item + self.model_parameters.generate_max_tokens + 1, #+1 for <SEP> 
            n_embd =  model_parameters.embedding_size, 
            n_layer = model_parameters.transformer_blocks, 
            n_head = model_parameters.attention_heads, 
        )
        self.num_items = data_parameters.num_items 
        self.gpt = TFGPT2LMHeadModel(gpt_config) 
        pass

    @classmethod
    def get_model_config_class(cls) -> Type[RLGPT2RecConfig]:
        return RLGPT2RecConfig

    def fit_biases(self, train_users):
        self.tokenizer.assign(train_users)
        self.tokenizer.build_index()

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        seq = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length + self.model_parameters.generate_max_tokens + 1), 'int64')
        input_ids = tf.zeros_like(seq)
        return [seq, seq, input_ids]

    def call(self, inputs, **kwargs):
        input_seq = inputs[1] 
        position_ids = inputs[2]
        attention_mask, gpt_input, gpt_labels = self.get_gpt_inputs(input_seq) 
        result = self.gpt(input_ids=gpt_input, position_ids=position_ids, attention_mask=attention_mask)
        recommendation_logits = result.logits[:, -self.model_parameters.generate_max_tokens -1: -1, :]
        item_logprobs = tf.nn.log_softmax(recommendation_logits, axis=-1)
        recommendation_ids = input_seq[:,-self.model_parameters.generate_max_tokens:]
        recommendation_logprobs = tf.gather(item_logprobs, recommendation_ids, batch_dims=2, axis=-1)
        loss = -tf.reduce_mean(recommendation_logprobs)
        return loss 

    def get_gpt_inputs(self, input_seq):
        tokens = self.tokenizer(input_seq, self.data_parameters.batch_size, self.data_parameters.sequence_length  + self.model_parameters.generate_max_tokens + 1)
        attention_mask = tf.cast((tokens != -100), 'float32')
        tokens = tf.nn.relu(tokens)
        gpt_input = tokens 
        gpt_labels = tokens
        return attention_mask,gpt_input,gpt_labels



    def score_all_items(self, inputs): 
        pass