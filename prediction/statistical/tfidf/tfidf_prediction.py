import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import SGDRegressor, Lasso, LassoLars
from sklearn.model_selection import train_test_split
import h5py

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
if not os.path.exists("x.h5"):
    print(1)
    train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
    text = train_data["TextOfVacancy"].values.astype('U')
    tfidf = CountVectorizer(min_df=(30 / len(text)), max_df=(len(text) * 0.9), binary=True)
    vect_text = tfidf.fit_transform(text)
    names = list(tfidf.get_feature_names())
    pd.DataFrame(names).to_csv("names.csv", index=False, encoding="utf-8")
    x = hstack([vect_text, train_data[["City", "Exp", "EmploymentType", "WorkHours", "job_start"]].as_matrix()])
    del text
    del train_data
    h5f = h5py.File("x.h5", 'w')
    h5f.create_dataset('dataset_1', data=x.toarray())
    h5f.close()
    del x
names = list(pd.read_csv("names.csv")["0"])
y = np.log(pd.read_csv(data_path + "down_y_train_normalized.csv").as_matrix()) - 8
words = {}
n = 10000
h5f = h5py.File("x.h5", 'r')
for iteration in range(800):
    print(iteration)
    indexies = np.arange(len(y))
    np.random.shuffle(indexies)
    indexies = indexies[:n]
    indexies = np.sort(indexies)
    X_train = h5f['dataset_1'][list(indexies)]
    y_train = y[indexies]
    clf = LassoLars(alpha=0.0005)
    clf.fit(X_train, y_train)
    for i in range(len(names)):
        if np.abs(clf.coef_[i]) > 0.00001:
            if names[i] in words:
                words.update({names[i]:words[names[i]]+clf.coef_[i]})
            else:
                words.update({names[i]: 0})
pd.DataFrame.from_dict(words, orient="index").reset_index().to_csv("words_freq.csv", index=False, encoding="utf-8")