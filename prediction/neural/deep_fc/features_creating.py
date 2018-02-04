import pandas as pd
import os
import h5py
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
y_train = pd.read_csv(data_path + "down_y_train_normalized.csv").as_matrix()
y_test = pd.read_csv(data_path + "down_y_test_normalized.csv").as_matrix()
train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
test_data = pd.read_csv(data_path + "down_x_test_normalized.csv")
if not os.path.exists("data\\train_x1.h5"):
    print(1223)
    labels = ["TitleOfVacancy", "TextOfVacancy"]
    params = [[1000, 50000, 50, (1, 1)], [1000, 50000, 50, (1, 1)]]
    for j in range(1, 3):
        if not os.path.exists("data\\train_x"+str(j)+".h5"):
            text = train_data[labels[j-1]].values.astype('U')
            n_feat, batch_size, min_df, ngr = params[j-1]
            tfidf = CountVectorizer(min_df=min_df / len(text), max_features=n_feat, binary=True, ngram_range=ngr)
            tfidf.fit(text)
            h5f = h5py.File("data\\train_x"+str(j)+".h5", 'w')
            dset = h5f.create_dataset('dataset_1', (len(text), n_feat))
            for i in range(1, int(len(text) / batch_size) + 1):
                print(i)
                dset[(i - 1) * batch_size:i * batch_size, :] = tfidf.transform(
                    text[(i - 1) * batch_size:i * batch_size]).toarray()
            dset[-(len(text) % batch_size):, :] = tfidf.transform(text[-(len(text) % batch_size):]).toarray()
            h5f.close()

            h5f = h5py.File("data\\test_x"+str(j)+".h5", 'w')
            h5f.create_dataset('dataset_1', data=tfidf.transform(test_data[labels[j-1]].values.astype('U')).toarray())
            h5f.close()

if not os.path.exists("data\\train_x3.h5"):
    print(2)
    cities = train_data["City"].values.astype('U')
    tfidf = CountVectorizer(max_features=50, binary=True, ngram_range=(1, 1))
    train_x3 = tfidf.fit_transform(cities)
    h5f = h5py.File("data\\train_x3.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x3.toarray())
    h5f.close()

    h5f = h5py.File("data\\test_x3.h5", 'w')
    test_x3 = tfidf.transform(test_data["City"].values.astype('U')).toarray()
    h5f.create_dataset('dataset_1', data=test_x3)
    h5f.close()

if not os.path.exists("data\\train_x4.h5"):
    print(3)
    name = train_data["NameOfCompany"].values.astype('U')
    tfidf = CountVectorizer(min_df=50 / len(name), binary=True, ngram_range=(1, 2))
    train_x4 = tfidf.fit_transform(name)
    print(3)
    h5f = h5py.File("data\\train_x4.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x4.toarray())
    h5f.close()
    h5f = h5py.File("data\\test_x4.h5", 'w')
    h5f.create_dataset('dataset_1', data=tfidf.transform(test_data["NameOfCompany"].values.astype('U')).toarray())
    h5f.close()

if not os.path.exists("data\\train_x5.h5"):
    train_x5 = train_data[["Exp", "EmploymentType", "WorkHours"]].as_matrix()
    h5f = h5py.File("data\\train_x5.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x5)
    h5f.close()

    test_x5 = test_data[["Exp", "EmploymentType", "WorkHours"]].as_matrix()
    h5f = h5py.File("data\\test_x5.h5", 'w')
    h5f.create_dataset('dataset_1', data=test_x5)
    h5f.close()


if not os.path.exists("data\\train_y.h5"):
    print(45465)
    min_y = np.min(y_train)
    max_y = np.max(y_train)
    y_train = (y_train)/(max_y)
    y_test = (y_test)/(max_y)

    h5f = h5py.File("data\\train_y.h5", 'w')
    h5f.create_dataset('dataset_1', data=y_train)
    h5f.close()

    h5f = h5py.File("data\\test_y.h5", 'w')
    h5f.create_dataset('dataset_1', data=y_test)
    h5f.close()

if not os.path.exists("data\\train_clusters.h5"):
    train_x5 = train_data["cluster"].as_matrix()
    h5f = h5py.File("data\\train_clusters.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x5)
    h5f.close()

    test_x5 = test_data["cluster"].as_matrix()
    h5f = h5py.File("data\\test_clusters.h5", 'w')
    h5f.create_dataset('dataset_1', data=test_x5)
    h5f.close()
