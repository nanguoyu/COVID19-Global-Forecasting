"""
@File : testLSTMModel.py
@Author: Dong Wang
@Date : 2020/4/23
"""
from WorkFlow.core.DataLoader import DataLoader
from WorkFlow.core.model import LSTMModel
import os

split = 0.8
epoch = 100
batch_size = 10
save_dir = "saved_models"
file = 'D:/Dropbox/Project/code/Python/COVID19-Global-Forecasting/data/train_flight.csv'
pretrained_weights = os.path.join(save_dir, 'COVID19_LSTM.h5')

print("[Test]:Load data from ", file)
data_loader = DataLoader(filename=file, split=split)
trainX, trainY = data_loader.get_train_data(seq_len=7, normalise=True)
valX, valY = data_loader.get_evaluate_data(seq_len=7, normalise=True)
print("[Test]: trainX shape", trainX.shape, "trainY shape", trainY.shape)
print("[Test]: valX shape", valX.shape, "valY shape", valY.shape)

lstm = LSTMModel()
lstm.buildModel(pretrained_weights=pretrained_weights)
lstm.train(trainX, trainY, xval=valX, yval=valY, epochs=epoch, batch_size=batch_size, save_dir=save_dir)
print("[Test]:Done")
