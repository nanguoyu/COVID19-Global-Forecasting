"""
@File : testDataLoader.py
@Author: Dong Wang
@Date : 2020/4/21
"""
from WorkFlow.core.DataLoader import DataLoader

file = 'D:/Dropbox/Project/code/Python/COVID19-Global-Forecasting/data/train_flight.csv'
split = 0.8
print("[Test]:Load data from ", file)
data_loader = DataLoader(filename=file, split=split)
trainX, trainY = data_loader.get_train_data(seq_len=7, normalise=False)
valX, valY = data_loader.get_evaluate_data(seq_len=7, normalise=False)
print("[Test]: trainX shape", trainX.shape, "trainY shape", trainY.shape)
print("[Test]: valX shape", valX.shape, "valY shape", valY.shape)
print("[Test]:Done")
