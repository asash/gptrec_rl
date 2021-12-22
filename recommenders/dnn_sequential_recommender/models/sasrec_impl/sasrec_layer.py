#This is a port of original SASRec code
#Taken from https://github.com/kang205/SASRec,
#Ported to python3/tensorflow2
#Converted to Keras layer.


import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.python.keras import activations

from .modules import embedding, multihead_attention, normalize, \
    feedforward


class SASRecLayer(layers.Layer):
    def __init__(self, itemnum,
                 hidden_units=50,
                 l2_emb=0.0,
                 maxlen=50,
                 dropout_rate=0.5,
                 num_blocks=2,
                 num_heads=1,
                 activation='linear',
                 **kwargs):

        super().__init__(**kwargs)
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.l2_emb = l2_emb
        self.maxlen = maxlen
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.activation = activations.get(activation)


    def call(self, input_seq, **kwargs):
        is_training = kwargs['training']
        if is_training is None:
            is_training = K.learning_phase()
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), dtype=tf.float32), -1)

        with tf.compat.v1.variable_scope("SASRec"):
            # sequence embedding, item embedding table
            seq, item_emb_table = embedding(input_seq,
                                                 vocab_size=self.itemnum + 1,
                                                 num_units=self.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=self.l2_emb,
                                                 scope="input_embeddings",
                                                 reuse=tf.compat.v1.AUTO_REUSE,
                                                 with_t=True,
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input=input_seq)[1]), 0), [tf.shape(input=input_seq)[0], 1]),
                vocab_size=self.maxlen,
                num_units=self.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="dec_pos",
                reuse=tf.compat.v1.AUTO_REUSE,
                with_t=True
            )
            seq += t

            # Dropout
            seq = tf.compat.v1.layers.dropout(seq,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(value=is_training))
            seq *= mask

            # Build blocks

            for i in range(self.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    seq = multihead_attention(queries=normalize(seq),
                                                   keys=seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=is_training,
                                                   causality=True,
                                                   reuse=tf.compat.v1.AUTO_REUSE,
                                                   scope="self_attention")

                    # Feed forward
                    seq = feedforward(normalize(seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=is_training,
                                           reuse=tf.compat.v1.AUTO_REUSE
                                      )
                    seq *= mask
            seq = normalize(seq)
            seq = tf.reduce_sum(seq, axis=1)
            seq = normalize(seq)
            output = seq @ tf.transpose(item_emb_table)
            output = self.activation(output)
            return output[:,1:]