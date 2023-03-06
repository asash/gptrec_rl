import tensorflow as tf
from .centroid_assignment_strategies.centroid_strategy import CentroidAssignmentStragety
from .centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy

def get_codes_strategy(codes_strategy, item_code_bytes, num_items) -> CentroidAssignmentStragety:
    if codes_strategy == "svd":
        return SVDAssignmentStrategy(item_code_bytes, num_items)
        
class ItemCodeLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, pq_m, num_items, sequence_length, codes_strategy):
        super().__init__()
        self.sub_embedding_size = embedding_size // pq_m
        self.item_code_bytes = embedding_size // self.sub_embedding_size
        item_initializer = tf.zeros_initializer()
        self.item_codes = tf.Variable(item_initializer((num_items + 1, self.item_code_bytes), dtype='uint8'), 
                                      trainable=False, name="ItemCodes/codes")

        centroid_initializer = tf.random_uniform_initializer()
        self.centroids = tf.Variable(centroid_initializer(shape=(self.item_code_bytes, 256,
                                                                 self.sub_embedding_size)),
                                     name="ItemCodes/centroids")
        self.item_codes_strategy = get_codes_strategy(codes_strategy, self.item_code_bytes, num_items)
        self.sequence_length = sequence_length

    def assign_codes(self, train_users):
        codes = self.item_codes_strategy.assign(train_users)
        self.item_codes.assign(codes)

    def call(self, input_ids, batch_size):
        input_codes = tf.stop_gradient(tf.cast(tf.gather(self.item_codes, input_ids), 'int32'))
        code_byte_indices = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, self.item_code_bytes), 0), 0), [batch_size, self.sequence_length,1])
        n_sub_embeddings = batch_size * self.sequence_length * self.item_code_bytes
        code_byte_indices_reshaped = tf.reshape(code_byte_indices, (n_sub_embeddings, ))
        input_codes_reshaped = tf.reshape(input_codes, (n_sub_embeddings,))
        indices = tf.stack([code_byte_indices_reshaped, input_codes_reshaped], axis=-1)
        input_sub_embeddings_reshaped = tf.gather_nd(self.centroids, indices)
        result = tf.reshape(input_sub_embeddings_reshaped,[batch_size, self.sequence_length, self.item_code_bytes * self.sub_embedding_size] )
        return result

