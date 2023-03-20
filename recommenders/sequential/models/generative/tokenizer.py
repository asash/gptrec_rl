from __future__ import annotations
import numpy as np

import numpy.typing as npt
import tensorflow as tf

class Tokenizer(object):
    def __init__(self, tokens_per_item, values_per_dimension, num_items) -> None:
        self.tokens_per_item = tokens_per_item
        self.num_items = num_items
        self.values_per_dimension = values_per_dimension 
        self.vocab_size = tokens_per_item * values_per_dimension 
        self.vocabulary = tf.constant(np.zeros((num_items, tokens_per_item), dtype='int32'))
        self.index = {}
    
    def assign(self, train_users):
        raise NotImplementedError()
    
    def build_index(self):
        for i in range(len(self.vocabulary)):
            item_tokens = tuple(self.vocabulary[i].numpy())
            self.index[item_tokens] = i
    
    def decode(self, batch):
        result = [] 
        for i in range(len(batch)):
           row = tuple(batch[i].numpy())
           item = self.index.get(row, self.num_items) # if incorrect sequence was generated return pad token
           result.append(item)
        return tf.constant(np.array(result))


    def tokenize(self, batch: npt.ArrayLike):
        return tf.reshape(tf.gather(self.vocabulary, batch), (batch.shape[0], batch.shape[1]*self.tokens_per_item))
    
