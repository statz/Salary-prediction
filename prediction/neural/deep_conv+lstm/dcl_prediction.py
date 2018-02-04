import pandas as pd
import os
import numpy as np
import h5py
from keras.layers import LSTM
from keras.models import Model
from matplotlib import pyplot
from keras.callbacks import History
from keras.layers import Dense, Dropout
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"

h5f = h5py.File(os.getcwd() + "\\data\\" + "down_vectors_repr.h5", 'r')
embedding_matrix = h5f['dataset_1'][:]
h5f.close()

val_len = 10000
n_train = 760000
val_x1 = h5py.File("data\\train_x1.h5", 'r')['dataset_1'][:val_len]
val_x2 = h5py.File("data\\train_x2.h5", 'r')['dataset_1'][:val_len]
val_x3 = h5py.File("data\\train_x3.h5", 'r')['dataset_1'][:val_len]
val_x4 = h5py.File("data\\train_x4.h5", 'r')['dataset_1'][:val_len]
val_x5 = h5py.File("data\\train_x5.h5", 'r')['dataset_1'][:val_len]
# y_val = h5py.File("data\\train_y.h5", 'r')['dataset_1'][:val_len]

train_x1 = HDF5Matrix('data\\train_x1.h5', 'dataset_1', start=val_len, end=n_train)
train_x2 = HDF5Matrix('data\\train_x2.h5', 'dataset_1', start=val_len, end=n_train)
train_x3 = HDF5Matrix('data\\train_x3.h5', 'dataset_1', start=val_len, end=n_train)
train_x4 = HDF5Matrix('data\\train_x4.h5', 'dataset_1', start=val_len, end=n_train)
train_x5 = HDF5Matrix('data\\train_x5.h5', 'dataset_1', start=val_len, end=n_train)
# y_train = h5py.File("data\\train_y.h5", 'r')['dataset_1'][val_len:n_train]

val_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()[:val_len]
train_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()[val_len:n_train]
min_y = train_y.min()
max_y = train_y.max()
train_y = (train_y) /1000 #/ (max_y - min_y)
val_y = (val_y) /1000#/ (max_y - min_y)
from keras.layers import Embedding

print('Training model.')

history = History()

input1 = Input(shape=(train_x1.shape[1],))
embedding_layer = Embedding(embedding_matrix.shape[0], 50, weights=[embedding_matrix], input_length=4,
                            trainable=False)
embedded_sequences = embedding_layer(input1)
x1 = LSTM(50, dropout=0.4)(embedded_sequences)
input4 = Input(shape=(train_x4.shape[1],))
x1 = concatenate([x1, input4])
x1 = Dense(10, activation='tanh')(x1)
x1 = Dropout(0.1)(x1)

input2 = Input(shape=(train_x2.shape[1],))
input3 = Input(shape=(train_x3.shape[1],))
input5 = Input(shape=(train_x5.shape[1],))
x2 = Dense(1500, activation='tanh')(input2)
x2 = Dropout(0.55)(x2)
x2 = Dense(500, activation='tanh')(x2)
x2 = Dropout(0.45)(x2)
x = concatenate([x1, x2])
x = Dense(200, activation='tanh')(x)
x = Dropout(0.4)(x)
x3 = Dense(1, activation='tanh')(input3)
x = concatenate([x, x3, input5])
x = Dense(100, activation='tanh')(x)
x = Dropout(0.3)(x)
x = Dense(10, activation='tanh')(x)
x = Dropout(0.15)(x)
preds = Dense(1, activation='relu')(x)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')

model = Model([input1, input2, input3, input4, input5], preds)
opt = Adam()
model.compile(loss='mean_absolute_percentage_error',
              optimizer=opt,
              metrics=['mean_absolute_percentage_error', "mean_absolute_error"])

model.fit([train_x1, train_x2, train_x3, train_x4, train_x5],
          train_y, batch_size=1000, epochs=600, shuffle='batch',
          validation_data=([[val_x1, val_x2, val_x3, val_x4, val_x5], val_y]), callbacks=[es])
model.save(os.getcwd() + "\\data\\" + "model.h5")