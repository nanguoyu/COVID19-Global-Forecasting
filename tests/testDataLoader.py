"""
@File : testDataLoader.py
@Author: Dong Wang
@Date : 2020/4/21
"""
from WorkFlow.core.DataLoader import DataLoader

file = 'D:/Dropbox/Project/code/Python/COVID19-Global-Forecasting/data/train_flight.csv'
split = 0.8
data_loader = DataLoader(filename=file, split=split)
trainX, trainY = data_loader.get_train_data(seq_len=7, normalise=False)

print("trainX shape", trainX.shape, "trainY shape", trainY.shape)
