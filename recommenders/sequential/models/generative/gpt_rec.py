from __future__ import annotations

from typing import List, Type
import tensorflow as tf
from aprec.recommenders.sequential.models.generative.tokenizer import Tokenizer
from aprec.recommenders.sequential.models.generative.tokenizers.tokenizer_utils import get_tokenizer_class
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
from transformers import GPT2Config, TFGPT2LMHeadModel 

class GPT2RecConfig(SequentialModelConfig):
    def __init__(self,
                 embedding_size = 256, 
                 transformer_blocks = 3, 
                 attention_heads = 4, 
                 tokenizer='svd',
                 values_per_dim = 1024, 
                 tokens_per_item = 4, 
                 generate_top_k = 50, 
                 generate_top_p = 0.95,
                 generate_n_sequences = 50 
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

        
    def as_dict(self):
        return self.__dict__
    
    def get_model_architecture(self) -> Type[GPT2RecConfig]:
        return GPT2RecModel

class GPT2RecModel(SequentialRecsysModel):
    def __init__(self, model_parameters: GPT2RecConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: GPT2RecConfig
        self.tokenizer_class = get_tokenizer_class(model_parameters.tokenizer)
        self.tokenizer:Tokenizer = self.tokenizer_class(model_parameters.tokens_per_item, model_parameters.values_per_dim, data_parameters.num_items)
        gpt_config = GPT2Config(
            vocab_size = self.tokenizer.vocab_size,
            n_positions = data_parameters.sequence_length * model_parameters.tokens_per_item, 
            n_embd =  model_parameters.embedding_size, 
            n_layer = model_parameters.transformer_blocks, 
            n_head = model_parameters.attention_heads, 
            
        )
        self.num_items = data_parameters.num_items 
        self.gpt = TFGPT2LMHeadModel(gpt_config) 
        pass

    @classmethod
    def get_model_config_class(cls) -> Type[GPT2Config]:
        return GPT2Config

    def fit_biases(self, train_users):
        self.tokenizer.assign(train_users)
        self.tokenizer.build_index()

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        seq = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int64')
        return [seq]

    def call(self, inputs, **kwargs):
        tokens = self.tokenizer.tokenize(inputs[0])
        attention_mask = tf.cast((tokens != -100), 'float32')
        tokens = tf.nn.relu(tokens)
        gpt_input=tokens
        gpt_labels=tokens
        #gpt_input = tokens[:,:-1]
        #attention_mask = attention_mask[:,:-1]
        #gpt_labels = tokens[:,1:]
        result = self.gpt(input_ids=gpt_input, labels=gpt_labels, return_dict=True, attention_mask=attention_mask)
        return result.loss

   
    def score_all_items(self, inputs): 
        tokens = self.tokenizer.tokenize(inputs[0])
        attention_mask = tf.cast((tokens != -100), 'float32')
        tokens = tf.nn.relu(tokens)
        output = self.gpt.generate(
                tokens,
                do_sample=True, 
                max_new_tokens=self.model_parameters.tokens_per_item, 
                top_k=self.model_parameters.generate_top_k, 
                top_p=self.model_parameters.generate_top_p, 
                num_return_sequences=self.model_parameters.generate_n_sequences, 
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask
                )
        input_length = tokens.shape[-1]
        generated_sequences = output.sequences[:,input_length:]
        predicted_items = self.tokenizer.decode(generated_sequences)
        logits = tf.transpose(tf.stack(output.scores), [1, 0, 2])
        token_scores = tf.gather(logits, generated_sequences, batch_dims=2)
        seq_scores = tf.reduce_sum(token_scores, axis=-1)
        predicted_items =tf.reshape(predicted_items, (tokens.shape[0], self.model_parameters.generate_n_sequences))
        sample_nums = tf.cast(tf.tile(tf.expand_dims(tf.range(0, predicted_items.shape[0]), -1), [1, self.model_parameters.generate_n_sequences]), 'int64')
        index = tf.stack([sample_nums, predicted_items], axis=-1)
        shape = tf.constant([tokens.shape[0], self.data_parameters.num_items + 1], dtype='int64')
        seq_scores = tf.reshape(seq_scores, (tokens.shape[0], self.model_parameters.generate_n_sequences))
        result = tf.scatter_nd(index, seq_scores, shape)[:,:-1]
        print(tf.reduce_sum(result))
        return result