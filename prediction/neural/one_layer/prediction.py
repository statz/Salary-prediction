import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import h5py
from matplotlib import pyplot
from nltk.corpus import stopwords

n = -1
data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
y_train = pd.read_csv(data_path + "down_y_train_normalized.csv").as_matrix()
y_test = pd.read_csv(data_path + "down_y_test_normalized.csv").as_matrix()
mx = max(max(y_test), max(y_train))
mn = min(min(y_test), min(y_train))
y_train = (y_train) / (mx - mn)
if not os.path.exists("x1.h5"):
    print(1)
    train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
    text = train_data["combined_text"].values.astype('U')
    vocabluary = list(pd.read_csv("words_freq.csv")["name"])
    tfidf = CountVectorizer(min_df= 100 / len(text), binary=True, ngram_range=(1, 1)
                            ,stop_words=stopwords.words('russian') + stopwords.words('english'))
    vect_text_train = tfidf.fit_transform(text)
    train_x = hstack([vect_text_train, train_data[["Exp", "EmploymentType", "WorkHours", "job_start",
                                                   "москва", "спб", "mean_sal"]].as_matrix()])
    #train_x = vect_text_train
    print(train_x.shape)
    print(2)
    h5f = h5py.File("x1.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x.toarray())
    h5f.close()
    print(3)
print(4)
h5f = h5py.File("x1.h5", 'r')
train_x = h5f['dataset_1'][:]
if n > 0:
    indexies = np.arange(len(y_train))
    np.random.shuffle(indexies)
    indexies = indexies[:n]
    indexies = np.sort(indexies)
    print(5)
    train_x = train_x[[list(indexies)]]
    y_train = y_train[list(indexies)]
print(6)
train_x[:, -1] /= (mx - mn)


from keras.layers import Dropout
from keras.callbacks import History
from keras.layers import Dense, Input, concatenate
from keras.models import Model

history = History()
sequence_input = Input(shape=(train_x.shape[1]-7,))
x = Dense(1000, activation='tanh')(sequence_input)
x = Dropout(0.65)(x)
x = Dense(150, activation='tanh')(sequence_input)
x = Dropout(0.5)(x)
auxiliary_input = Input(shape=(7,), name='aux_input')
#y = Dense(10, activation='relu')(auxiliary_input)
#y = Dropout(0.2)(y)
x = concatenate([x, auxiliary_input])
x = Dense(100, activation='tanh')(x)
x = Dropout(0.3)(x)
x = Dense(10, activation='tanh')(x)
x = Dropout(0.1)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model([sequence_input,auxiliary_input], preds)
model.compile(loss='mean_absolute_percentage_error',
              optimizer='adam',
              metrics=['mean_absolute_percentage_error'])
model.fit([train_x[:, :-7], train_x[:, -7:]],
          y_train, batch_size=500, epochs=250, validation_split=0.05, callbacks=[history])

#model = Sequential()
#model.add(Dense(1500, input_dim=train_x.shape[1], activation='tanh', kernel_initializer='random_uniform',))
#model.add(Dropout(0.25))
#model.add(Dense(1, activation='sigmoid'))

#model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
#model.fit(train_x, y_train, epochs=100, batch_size=500, validation_split=0.1, callbacks=[history])
loss = history.history["val_loss"]
pyplot.plot(np.arange(len(loss)), loss)
pyplot.show()
data = model.get_weights()[0]
pd.DataFrame(data).to_csv("aaa.csv")
#print(0)
#test_data = pd.read_csv(data_path + "down_x_test_normalized.csv")
#text = test_data["TextOfVacancy"].values.astype('U')
#vect_text_test = tfidf.transform(text)
#test_x = hstack([vect_text_test, test_data[
#    ["Exp", "EmploymentType", "WorkHours", "job_start", "москва", "спб"]].as_matrix()]).toarray()
#print(1)
#
#pr = model.predict(test_x) * (mx - mn)
#s = 0
#for i in range(len(pr)):
#    s += np.abs((pr[i] - y_test[i]) / y_test[i])
#print(s / len(pr))
#