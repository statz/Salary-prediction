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

train_ind = pd.read_csv(os.getcwd()+"\\data\\"+'down_ind_train_x.csv').as_matrix()
embedding_matrix = pd.read_csv(os.getcwd()+"\\data\\"+'down_vectors_repr.csv').as_matrix()
train_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()
min_y = train_y.min()
max_y = train_y.max()
train_y = minmax_scale((train_y))
data = pd.read_csv(data_path + "down_x_train_normalized.csv")[["Exp", "EmploymentType", "WorkHours", "job_start",
                                                   "москва", "спб", "mean_sal"]].as_matrix()
data[:, -1] /= (max_y-min_y)
from keras.layers import Embedding

embedding_layer = Embedding(embedding_matrix.shape[0],
                            50,
                            weights=[embedding_matrix],
                            input_length=84,
                            trainable=False)

print('Training model.')
from keras.callbacks import History

history = History()
from keras.layers import Dense, Dropout

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(84,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(100, 50, activation='tanh')(embedded_sequences)
x = MaxPooling1D(30)(x)
x = Flatten()(x)
#x = Dropout(0.25)(x)
#x = LSTM(20, dropout=0.2)(x)
auxiliary_input = Input(shape=(7,), name='aux_input')
x = keras.layers.concatenate([x, auxiliary_input])
x = Dense(128, activation='tanh')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model([sequence_input,auxiliary_input], preds)
model.compile(loss='mean_absolute_percentage_error',
              optimizer='adamax',
              metrics=['mean_absolute_percentage_error'])

model.fit([train_ind, data],
          train_y, batch_size=500, epochs=50, validation_split=0.1, callbacks=[history])

#print("aaaa")
#test_ind = pd.read_csv(os.getcwd()+"\\data\\"+'down_ind_test_x.csv').as_matrix()
#test_y = pd.read_csv(data_path + "down_y_test.csv")["down"].as_matrix()
#print("sadasdas")
#pr = (max_y-min_y)*model.predict(test_ind)+min_y
#print("sdadafasdfasfasfsa")
#err = 0
#for i in range(len(pr)):
#    err += (np.abs(pr[i]-test_y[i]))/len(pr)
#print("euioegjregj")
#print(err)

