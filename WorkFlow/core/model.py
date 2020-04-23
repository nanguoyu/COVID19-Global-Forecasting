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

    def train(self, x, y, xval, yval, epochs, batch_size, save_dir):
        raise NotImplementedError

    def getModel(self):
        return self._model


class LSTMModel(Model):
    def __init__(self):
        super(LSTMModel, self).__init__()

    def train(self, x, y, xval, yval, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        save_fname = os.path.join(save_dir, 'COVID19_LSTM.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
        ]
        history = self._model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(xval, yval)
        )
        self._model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        # print('history.keys ', history.history.keys())
        loss = history.history['loss']
        # ['val_loss', 'val_accuracy', 'loss', 'accuracy']
        acc = history.history['accuracy']
        # make a figure
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # subplot loss
        # ax1 = fig.add_subplot(121)
        ax1.plot(loss, label='train_loss')
        ax1.legend(loc=4)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss on Training')

        # ax2 = fig.add_axes(122)
        ax2.plot(acc, label='train_acc')
        ax2.legend(loc=4)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Acc')
        ax2.set_title('Acc on Training')
        ax1.legend()
        fig.savefig('./loss_acc_fig.png')

    def buildModel(self, pretrained_weights=None):
        self._model.add(
            LSTM(256, activation="tanh", input_shape=(6, 2), return_sequences=True))
        self._model.add(Dropout(0.2))
        self._model.add(
            LSTM(512, activation="tanh", return_sequences=True))
        self._model.add(
            LSTM(256, activation="tanh", return_sequences=False))
        self._model.add(Dropout(0.2))
        # self._model.add(BatchNormalization())
        self._model.add(Dense(1, activation='linear'))

        if pretrained_weights:
            self._model.load_weights(pretrained_weights)
        learning_rate = 1e-5
        momentum = 0.8
        decay_rate = learning_rate / 64
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        self._model.compile(loss='mse', optimizer="adam",
                            metrics=['accuracy'])

    def evaluate(self):
        # TODO implement evaluate function
        pass
