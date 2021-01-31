from tempfile import NamedTemporaryFile
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Convolution1D, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Attention, Multiply, Dropout, BatchNormalization, LayerNormalization, \
    Add, Maximum
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import Sequence
import keras.losses


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
    def __init__(self, features_num, query_group_size, batch_size=100, n_layers=15, n_heads=10, epochs=10000,
                 attention = True,
                 early_stopping=40,
                 dropout = True):
        input = Input(shape=(query_group_size, features_num))
        x = input
        x = BatchNormalization()(x)
        for layer_size in range(n_layers):
            shortcut = x
            heads = []
            for head in range(n_heads):
                heads.append(Convolution1D(x.shape[-1], 1, activation='swish')(x))
            x = Maximum()(heads)
            x = Add()([x, shortcut])
            x = LayerNormalization()(x)
            x = Dropout(0.1)(x)
        if attention:
            output = Attention()([x, x])
            output = Multiply()([output, x])
        else:
            output = x

        output = Convolution1D(output.shape[-1], 1, activation="sigmoid")(output)
        if (dropout):
            output = Dropout(0.5)(output)
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
        with NamedTemporaryFile(suffix='.h5', prefix='best_model_') as tmp:
            es = EarlyStopping(monitor='val_ndcg_at_40', mode='max', patience=self.early_stopping, verbose=1)
            mc = ModelCheckpoint(tmp.name, monitor='val_ndcg_at_40', mode='max', verbose=1,
                                 save_best_only=True, save_weights_only=True)
            self.model.fit(data_generator, validation_data=val_generator, epochs=self.n_epochs, callbacks=[es, mc])
            self.model.load_weights(tmp.name)

    def predict(self, x):
        request = np.array(x)
        request = request.reshape(1, request.shape[0], request.shape[1])
        prediction =  self.model.predict(request)[0]
        return prediction[:,0]