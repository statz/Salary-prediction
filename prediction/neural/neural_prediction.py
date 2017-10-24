import pandas as pd
import os
import numpy as np
import h5py
import keras
from sklearn.preprocessing import minmax_scale
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"

train_ind = pd.read_csv(os.getcwd()+"\\data\\"+'up_ind_train_x.csv').as_matrix()
embedding_matrix = pd.read_csv(os.getcwd()+"\\data\\"+'up_vectors_repr.csv').as_matrix()
train_y = pd.read_csv(data_path + "up_y_train.csv")["up"].as_matrix()
min_y = train_y.min()
max_y = train_y.max()
train_y = minmax_scale((train_y))

from keras.layers import Embedding

embedding_layer = Embedding(embedding_matrix.shape[0],
                            50,
                            weights=[embedding_matrix],
                            input_length=84,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(84,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(300, 20, activation='sigmoid')(embedded_sequences)
x = MaxPooling1D(20)(x)
x = Flatten()(x)
x = LSTM(100, dropout=0.2)(x)
x = Dense(128, activation='sigmoid')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='mean_absolute_percentage_error',
              optimizer='adam',
              metrics=['mean_absolute_percentage_error'])

model.fit(train_ind, train_y,
          batch_size=128,
          epochs=1)
print("aaaa")
test_ind = pd.read_csv(os.getcwd()+"\\data\\"+'up_ind_test_x.csv').as_matrix()
test_y = pd.read_csv(data_path + "up_y_test.csv")["up"].as_matrix()
print("sadasdas")
pr = (max_y-min_y)*model.predict(test_ind)+min_y
print("sdadafasdfasfasfsa")
err = 0
for i in range(len(pr)):
    err += (np.abs(pr[i]-test_y[i]))/len(pr)
print("euioegjregj")
print(err)
