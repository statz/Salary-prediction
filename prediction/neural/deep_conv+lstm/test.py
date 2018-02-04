import h5py
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"

val_len = 10000
n_train = 20000

h5f = h5py.File("data\\train_x2.h5",'r')
val_x2 = h5f['dataset_1'][:val_len]
h5f.close()

h5f = h5py.File("data\\train_x3.h5",'r')
val_x3 = h5f['dataset_1'][:val_len]
h5f.close()

h5f = h5py.File("data\\train_x4.h5",'r')
val_x4 = h5f['dataset_1'][:val_len]
h5f.close()

h5f = h5py.File("data\\train_x2.h5",'r')
train_x2 = h5f['dataset_1'][val_len:n_train]
h5f.close()

h5f = h5py.File("data\\train_x3.h5",'r')
train_x3 = h5f['dataset_1'][val_len:n_train]
h5f.close()

h5f = h5py.File("data\\train_x4.h5",'r')
train_x4 = h5f['dataset_1'][val_len:n_train]
h5f.close()

val = np.concatenate((val_x2, val_x3, val_x4), axis=1)
train = np.concatenate((train_x2, train_x3, train_x4), axis=1)

val_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()[:val_len]
train_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()[val_len:n_train]
min_y = train_y.min()
max_y = train_y.max()
train_y = (train_y) /1000 #/ (max_y - min_y)
val_y = (val_y) /1000#/ (max_y - min_y)

regr = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1,
    max_depth=2, random_state=0, loss='ls')
regr = regr.fit(train, train_y)
pr = regr.predict(val)
error = 0
for i in range(len(pr)):
    error += np.abs(pr[i]-val_y[i])/val_y[i]
print(error/len(pr))