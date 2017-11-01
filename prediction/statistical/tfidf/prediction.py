import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import h5py
from matplotlib import pyplot
n = 200000
data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
y_train = pd.read_csv(data_path + "down_y_train_normalized.csv").as_matrix()
y_test = pd.read_csv(data_path + "down_y_test_normalized.csv").as_matrix()
mx = max(max(y_test), max(y_train))
mn = min(min(y_test), min(y_train))
y_train = (y_train) / (mx - mn)
if not os.path.exists("x1.h5"):
    print(1)
    train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
    text = train_data["TextOfVacancy"].values.astype('U')
    vocabluary = list(pd.read_csv("words_freq.csv")["name"])
    tfidf = CountVectorizer(min_df=(30 / len(text)), max_df=(len(text) * 0.90), max_features=2000, binary=True, ngram_range=(1, 2))
    vect_text_train = tfidf.fit_transform(text)
    train_x = hstack([vect_text_train, train_data[["Exp", "EmploymentType", "WorkHours", "job_start",
                                                   "москва", "спб"]].as_matrix()])
    print(2)
    h5f = h5py.File("x1.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x.toarray())
    h5f.close()
    print(3)
print(4)
indexies = np.arange(len(y_train))
np.random.shuffle(indexies)
indexies = indexies[:n]
indexies = np.sort(indexies)
h5f = h5py.File("x1.h5", 'r')
train_x = h5f['dataset_1'][list(indexies)]
y_train = y_train[:n]
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import History
history = History()

model = Sequential()
model.add(Dense(3000, input_dim=train_x.shape[1], activation='sigmoid'))
model.add(Dropout(0.95))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
model.fit(train_x, y_train, epochs=300, batch_size=400, validation_split=0.1, callbacks=[history])
loss = history.history["val_loss"]
pyplot.plot(np.arange(len(loss)), loss)
pyplot.show()
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