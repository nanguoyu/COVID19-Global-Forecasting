"""
@File : model.py
@Author: Dong Wang
@Date : 2020/4/21
"""
from . import np, pd
import os
import math
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.optimizers import SGD


class Model(object):
    def __init__(self):
        self._model = Sequential()

    def loadModel(self, model=None):
        self._model = load_model(model)

    def buildModel(self, pretrained_weights=None):
        raise NotImplementedError

    def train(self, x, y, epochs, batch_size, save_dir):
        raise NotImplementedError

    def getModel(self):
        return self._model


class LSTMModel(Model):
    def __init__(self):
        super(LSTMModel, self).__init__()

    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        save_fname = os.path.join(save_dir, 'COVID19_LSTM%s-e%s.h5' % (
            dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            TensorBoard(log_dir=save_dir+'/logs',
                        histogram_freq=0,
                        write_graph=True,
                        write_grads=True,
                        write_images=True,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None)
        ]
        self._model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self._model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)

    def buildModel(self, pretrained_weights=None):
        self._model.add(
            LSTM(100, activation="tanh", input_shape=(6, 2), return_sequences=True))
        self._model.add(Dropout(0.25))
        self._model.add(
            LSTM(100, activation="tanh", return_sequences=True))
        self._model.add(Dropout(0.25))
        self._model.add(
            LSTM(100, activation="tanh", return_sequences=True))
        self._model.add(BatchNormalization())
        self._model.add(Dense(1, activation='linear'))

        if pretrained_weights:
            self._model.load_weights(pretrained_weights)

    def evaluate(self):
        # TODO implement evaluate function
        pass


