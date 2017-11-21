import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import h5py
from keras.utils.io_utils import HDF5Matrix
from matplotlib import pyplot
from nltk.corpus import stopwords
import keras.backend as K
import tensorflow as tf



data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
y_train = pd.read_csv(data_path + "down_y_train_normalized.csv").as_matrix()
y_test = pd.read_csv(data_path + "down_y_test_normalized.csv").as_matrix()
mx = max(max(y_test), max(y_train))
mn = min(min(y_test), min(y_train))
y_train = (y_train) / (mx - mn)
train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
print(y_train.shape, train_data.shape)

if not os.path.exists("x1.h5"):
    print(1)
    text = train_data["combined_text"].values.astype('U')
    tfidf = CountVectorizer(min_df=100 / len(text), binary=True, ngram_range=(1, 1)
                            , stop_words=stopwords.words('russian') + stopwords.words('english'))
    train_x1 = tfidf.fit_transform(text)
    print(2)
    h5f = h5py.File("x1.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x1.toarray())
    h5f.close()
    print(3)

if not os.path.exists("x2.h5"):
    cities = train_data["City"].values.astype('U')
    tfidf = CountVectorizer(min_df=50/ len(cities), binary=True, ngram_range=(1, 1))
    train_x2 = tfidf.fit_transform(cities)
    print(2)
    h5f = h5py.File("x2.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x2.toarray())
    h5f.close()
    print(3)

if not os.path.exists("x3.h5"):
    name = train_data["NameOfCompany"].values.astype('U')
    tfidf = CountVectorizer(min_df=50/ len(name), binary=True, ngram_range=(1, 1))
    train_x3 = tfidf.fit_transform(name)
    print(2)
    h5f = h5py.File("x3.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x3.toarray())
    h5f.close()
    print(3)

train_x4 = train_data[["Exp", "EmploymentType", "WorkHours", "job_start",
                       "москва", "спб"]].as_matrix()
print(4)
train_x1 = HDF5Matrix('x1.h5', 'dataset_1')
train_x2 = HDF5Matrix('x2.h5', 'dataset_1')
train_x3 = HDF5Matrix('x3.h5', 'dataset_1')


from keras.layers import Dropout, BatchNormalization
from keras.callbacks import History
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

history = History()
input1 = Input(shape=(train_x1.shape[1],))
x = Dense(1000, activation='tanh')(input1)
x = Dropout(0.65)(x)

input2 = Input(shape=(train_x2.shape[1],))
input3 = Input(shape=(train_x3.shape[1],))
x = concatenate([x, input2, input3])
x = Dense(1000, activation='tanh')(x)
x = Dropout(0.5)(x)

input4 = Input(shape=(6,), name='aux_input')
x = concatenate([x, input4])
x = Dense(500, activation='tanh')(x)
x = Dropout(0.3)(x)

x = Dense(50, activation='tanh')(x)
x = Dropout(0.1)(x)

preds = Dense(1, activation='sigmoid')(x)

model = Model([input1, input2, input3, input4], preds)
opt = Adam()  # (lr = 0.1, decay = 0.1)
model.compile(loss='mean_absolute_percentage_error',
              optimizer=opt,
              metrics=['mean_absolute_percentage_error'])
model.fit([train_x1, train_x2, train_x3 ,train_x4],
          y_train, batch_size=500, epochs=250, callbacks=[history], shuffle=None)

# model = Sequential()
# model.add(Dense(1500, input_dim=train_x.shape[1], activation='tanh', kernel_initializer='random_uniform',))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
# model.fit(train_x, y_train, epochs=100, batch_size=500, validation_split=0.1, callbacks=[history])
loss = history.history["val_loss"]
pyplot.plot(np.arange(len(loss)), loss)
pyplot.show()
data = model.get_weights()[0]
pd.DataFrame(data).to_csv("aaa.csv")
# print(0)
# test_data = pd.read_csv(data_path + "down_x_test_normalized.csv")
# text = test_data["TextOfVacancy"].values.astype('U')
# vect_text_test = tfidf.transform(text)
# test_x = hstack([vect_text_test, test_data[
#    ["Exp", "EmploymentType", "WorkHours", "job_start", "москва", "спб"]].as_matrix()]).toarray()
# print(1)
#
# pr = model.predict(test_x) * (mx - mn)
# s = 0
# for i in range(len(pr)):
#    s += np.abs((pr[i] - y_test[i]) / y_test[i])
# print(s / len(pr))
#
