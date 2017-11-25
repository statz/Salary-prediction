import pandas as pd
import numpy as np
import h5py
from keras.utils.io_utils import HDF5Matrix
from matplotlib import pyplot

val_len = 10000

val_x1 = h5py.File("data\\train_x1.h5", 'r')['dataset_1'][:val_len]
val_x2 = h5py.File("data\\train_x2.h5", 'r')['dataset_1'][:val_len]
val_x3 = h5py.File("data\\train_x3.h5", 'r')['dataset_1'][:val_len]
val_x4 = h5py.File("data\\train_x4.h5", 'r')['dataset_1'][:val_len]
y_val = h5py.File("data\\train_y.h5", 'r')['dataset_1'][:val_len]

train_x1 = HDF5Matrix('data\\train_x1.h5', 'dataset_1', start=val_len)
train_x2 = HDF5Matrix('data\\train_x2.h5', 'dataset_1', start=val_len)
train_x3 = HDF5Matrix('data\\train_x3.h5', 'dataset_1', start=val_len)
train_x4 = HDF5Matrix('data\\train_x4.h5', 'dataset_1', start=val_len)
y_train = h5py.File("data\\train_y.h5", 'r')['dataset_1'][val_len:]

from keras.layers import Dropout, BatchNormalization
from keras.callbacks import History
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

history = History()
input1 = Input(shape=(train_x1.shape[1],))
input2 = Input(shape=(train_x2.shape[1],))
input3 = Input(shape=(train_x3.shape[1],))
input4 = Input(shape=(train_x4.shape[1],))
x = concatenate([input1, input2, input3, input4])
x = Dense(3000, activation='tanh')(x)
x = Dropout(0.75)(x)
x = Dense(1500, activation='tanh')(x)
x = Dropout(0.65)(x)
x = Dense(750, activation='tanh')(x)
x = Dropout(0.55)(x)
x = Dense(200, activation='tanh')(x)
x = Dropout(0.35)(x)
x = Dense(100, activation='tanh')(x)
x = Dropout(0.25)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model([input1, input2, input3, input4], preds)
opt = Adam(lr=0.00005)
model.compile(loss='mean_absolute_percentage_error',
              optimizer=opt,
              metrics=['mean_absolute_percentage_error'])
model.fit([train_x1, train_x2, train_x3, train_x4],
          y_train, batch_size=500, epochs=400, callbacks=[history], shuffle='batch',
          validation_data=([[val_x1, val_x2, val_x3, val_x4], y_val]))

loss = history.history["val_loss"]
pyplot.plot(np.arange(len(loss)), loss)
pyplot.show()
data = model.get_weights()[0]
pd.DataFrame(data).to_csv("aaa.csv")
