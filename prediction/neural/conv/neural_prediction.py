import pandas as pd
import os
import numpy as np
import h5py
import keras
from sklearn.preprocessing import minmax_scale
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, BatchNormalization
from keras.models import Model
from matplotlib import pyplot
from keras.callbacks import History
from keras.layers import Dense, Dropout

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
N = 100000
train_ind = pd.read_csv(os.getcwd() + "\\data\\" + 'down_ind_train_x.csv')[:N].as_matrix()
h5f = h5py.File(os.getcwd() + "\\data\\" + "down_vectors_repr.h5", 'r')
embedding_matrix = h5f['dataset_1'][:]
h5f.close()

train_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()[:N]
min_y = train_y.min()
max_y = train_y.max()
train_y = (train_y) / (max_y - min_y)
data = pd.read_csv(data_path + "down_x_train.csv")[["Exp", "EmploymentType", "WorkHours"]][:N].as_matrix()
#data[:, -1] /= (max_y - min_y)
from keras.layers import Embedding

embedding_layer = Embedding(embedding_matrix.shape[0], 50, weights=[embedding_matrix], input_length=150,
                            trainable=False)

print('Training model.')

history = History()

sequence_input = Input(shape=(150,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x1 = Conv1D(filters=32, kernel_size=50,strides=20, activation='relu', padding="same")(embedded_sequences)
x1 = Flatten()(x1)
x1 = Conv1D(filters=32, kernel_size=50,strides=20, activation='relu', padding="same")(x1)
x1 = Flatten()(x1)
#x = MaxPooling1D(2)(x)
#x = BatchNormalization()(x)
#x =  keras.layers.concatenate([x1,x2,x3])
x = Dropout(0.65)(x1)
x = Dense(400, activation='tanh')(x)
x = Dropout(0.65)(x)
# x = Dropout(0.25)(x)
# x = LSTM(200, dropout=0.7)(x)
auxiliary_input = Input(shape=(3,), name='aux_input')
x = keras.layers.concatenate([x, auxiliary_input])
x = Dense(200, activation='tanh')(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='tanh')(x)
x = Dropout(0.35)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model([sequence_input, auxiliary_input], preds)
model.compile(loss='mean_absolute_percentage_error',
              optimizer='adam',
              metrics=['mean_absolute_percentage_error'])

model.fit([train_ind, data],
          train_y, batch_size=500, epochs=1000, validation_split=0.1, callbacks=[history])
loss = history.history["val_loss"]
pyplot.plot(np.arange(len(loss)), loss)
pyplot.show()
