from tensorflow.keras import layers
from recommenders.dnn_sequential_recommender.models.sasrec_impl.sasrec_layer import SASRecLayer
from recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from keras.models import Model

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
class SASRec(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, l2_emb=0.0, dropout_rate=0.5, num_blocks=2, num_heads=1):
        super().__init__(output_layer_activation, embedding_size, max_history_len, )
        self.l2_emb = l2_emb
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads

    def get_model(self):
        sasrec_layer = SASRecLayer(itemnum=self.num_items,
                                   hidden_units=self.embedding_size,
                                   maxlen=self.max_history_length,
                                   l2_emb=self.l2_emb,
                                   dropout_rate=self.dropout_rate,
                                   num_blocks = self.num_blocks,
                                   num_heads=self.num_heads,
                                   activation=self.output_layer_activation
                                   )
        input = layers.Input(shape=(self.max_history_length))
        output = sasrec_layer(input)
        return Model([input], output)