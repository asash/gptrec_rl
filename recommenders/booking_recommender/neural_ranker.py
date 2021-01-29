import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Convolution1D, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Attention, Multiply
from tensorflow.python.keras.utils.data_utils import Sequence

from aprec.recommenders.losses.lambdarank import LambdaRankLoss
from aprec.recommenders.metrics.ndcg import KerasNDCG


class BatchGenerator(Sequence):
    def __init__(self, x, y, query_group_size, batch_size=100):
        self.n_samples = len(y) // query_group_size
        self.n_batches = self.n_samples // batch_size
        self.batch_size = batch_size
        self.x = np.array(x).reshape(self.n_batches,  self.batch_size,query_group_size, x.shape[-1]).astype('float32')
        self.y = np.array(y).reshape(self.n_batches, self.batch_size, query_group_size,).astype('float32')
        self.query_group_size = self.n_samples

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        return [self.x[idx]], self.y[idx]

class NeuralRanker(object):
    def __init__(self, features_num, query_group_size, batch_size=100, layer_sizes=(30,20,10,5), epochs=10000,
                 attention = True,
                 early_stopping=40):
        input = Input(shape=(query_group_size, features_num))
        x = input
        for layer_size in layer_sizes:
            x = Convolution1D(layer_size, 1, activation='swish')(x)
        if attention:
            output = Attention()([x, x])
            output = Multiply()([output, x])
        else:
            output = x
        output = Convolution1D(1, 1, activation='linear')(output)
        self.model = Model(inputs=[input], outputs=output)
        loss = LambdaRankLoss(n_items=query_group_size, batch_size=batch_size, ndcg_at=40)
        metrics = [KerasNDCG(40)]
        self.model.compile(optimizer='adam', metrics=metrics, loss=loss)
        self.query_group_size = query_group_size
        self.batch_size = batch_size
        self.n_epochs = epochs
        self.early_stopping = early_stopping

    def fit(self, x, y, val_x, val_y):
        data_generator = BatchGenerator(x, y, self.query_group_size, self.batch_size)
        val_generator = BatchGenerator(val_x, val_y, self.query_group_size, self.batch_size)
        es = EarlyStopping(monitor='val_ndcg_at_40', mode='max', patience=self.early_stopping, verbose=1)
        self.model.fit(data_generator, validation_data=val_generator, epochs=self.n_epochs, callbacks=[es])

    def predict(self, x):
        request = np.array(x)
        request = request.reshape(1, request.shape[0], request.shape[1])
        prediction =  self.model.predict(request)[0]
        return prediction[:,0]