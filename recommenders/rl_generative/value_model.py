from typing import List
import tensorflow as tf
from torch import Tensor
from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig, RLGPT2RecModel
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters


class ValueModel(RLGPT2RecModel):
    def __init__(self, model_parameters: RLGPT2RecConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs) 
        self.value_head = tf.keras.layers.Dense(1, name='value_head')

    def get_dummy_inputs(self):
        seq = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length + self.model_parameters.generate_max_tokens + 1), 'int64')
        return seq

    
    def call(self, *args, **kwargs):
        gpt_outputs = self.gpt(*args, **kwargs, output_hidden_states=True).hidden_states[-1]
        return tf.squeeze(self.value_head(gpt_outputs))