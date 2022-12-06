import tensorflow.keras.layers as layers
from  tensorflow.keras.models import Model
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialRecsysModelBuilder

#https://dl.acm.org/doi/abs/10.1145/3159652.3159656
#This is a simplified version of Caser model, which doesn't use user embeddings
#We assume that user embedding is not available for in the sequential recommendation case
class Caser(SequentialRecsysModelBuilder):
    def __init__(self,
                 output_layer_activation='linear', embedding_size=64, max_history_len=64,
                 n_vertical_filters=4, n_horizontal_filters=16,
                 dropout_ratio=0.5, activation='relu'):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.n_vertical_filters = n_vertical_filters
        self.n_horizontal_filters = n_horizontal_filters
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.model_type = "Caser"

    def get_model(self):
        input = layers.Input(shape=(self.max_history_length))
        model_inputs = [input]
        x = layers.Embedding(self.num_items + 1, self.embedding_size, dtype='float32')(input)
        x = layers.Reshape(target_shape=(self.max_history_length, self.embedding_size, 1))(x)
        vertical = layers.Convolution2D(self.n_vertical_filters, kernel_size=(self.max_history_length, 1),
                                        activation=self.activation)(x)
        vertical = layers.Flatten()(vertical)
        horizontals = []
        for i in range(self.max_history_length):
            horizontal_conv_size = i + 1
            horizontal_convolution = layers.Convolution2D(self.n_horizontal_filters,
                                                          kernel_size=(horizontal_conv_size,
                                                                       self.embedding_size), strides=(1, 1),
                                                          activation=self.activation)(x)
            pooled_convolution = layers.MaxPool2D(pool_size=(self.max_history_length - horizontal_conv_size + 1, 1)) \
                (horizontal_convolution)
            pooled_convolution = layers.Flatten()(pooled_convolution)
            horizontals.append(pooled_convolution)
        x = layers.Concatenate()([vertical] + horizontals)
        x = layers.Dropout(self.dropout_ratio)(x)
        x = layers.Dense(self.embedding_size, activation=self.activation)(x)

        output = layers.Dense(self.num_items, activation=self.output_layer_activation)(x)
        model = Model(model_inputs, outputs=output)
        return model
