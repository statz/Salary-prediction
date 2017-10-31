import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"

train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")[:200000]
y_train = pd.read_csv(data_path + "down_y_train_normalized.csv")[:200000].as_matrix()
y_test = pd.read_csv(data_path + "down_y_test_normalized.csv").as_matrix()
mx = max(max(y_test), max(y_train))
mn = min(min(y_test), min(y_train))
y_train = (y_train) / (mx - mn)
text = train_data["TextOfVacancy"].values.astype('U')
vocabluary = list(pd.read_csv("words_freq.csv")["name"])
tfidf = CountVectorizer(vocabulary=vocabluary, binary=True)
vect_text_train = tfidf.fit_transform(text)
train_x = hstack([vect_text_train, train_data[["City", "Exp", "EmploymentType", "WorkHours", "job_start",
                                               "москва", "спб"]].as_matrix()]).toarray()
print(train_x.shape)
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(60, input_dim=train_x.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
model.fit(train_x, y_train, epochs=15, batch_size=128)

print(0)
test_data = pd.read_csv(data_path + "down_x_test_normalized.csv")
text = test_data["TextOfVacancy"].values.astype('U')
vect_text_test = tfidf.transform(text)
test_x = hstack([vect_text_test, test_data[
    ["City", "Exp", "EmploymentType", "WorkHours", "job_start", "москва", "спб"]].as_matrix()]).toarray()
print(1)

pr = model.predict(test_x) * (mx - mn)
s = 0
for i in range(len(pr)):
    s += np.abs((pr[i] - y_test[i]) / y_test[i])
print(s / len(pr))
