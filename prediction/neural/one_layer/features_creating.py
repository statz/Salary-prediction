import pandas as pd
import os
import h5py
from sklearn.feature_extraction.text import CountVectorizer

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
y_train = pd.read_csv(data_path + "down_y_train_normalized.csv").as_matrix()
y_test = pd.read_csv(data_path + "down_y_test_normalized.csv").as_matrix()
train_data = pd.read_csv(data_path + "down_x_train_normalized.csv")
test_data = pd.read_csv(data_path + "down_x_test_normalized.csv")

if not os.path.exists("data\\train_x1.h5"):
    text = train_data["combined_text"].values.astype('U')
    n_feat = 10000
    batch_size = 20000
    tfidf = CountVectorizer(min_df=200 / len(text), max_df=0.5, max_features=n_feat, binary=True, ngram_range=(1, 2))
    tfidf.fit(text)
    h5f = h5py.File("data\\train_x1.h5", 'w')
    dset = h5f.create_dataset('dataset_1', (len(text), n_feat))
    for i in range(1, int(len(text) / batch_size) + 1):
        print(i)
        dset[(i - 1) * batch_size:i * batch_size, :] = tfidf.transform(
            text[(i - 1) * batch_size:i * batch_size]).toarray()
    dset[-(len(text) % batch_size):, :] = tfidf.transform(text[-(len(text) % batch_size):]).toarray()
    h5f.close()

    h5f = h5py.File("data\\test_x1.h5", 'w')
    h5f.create_dataset('dataset_1', data=tfidf.transform(test_data["combined_text"].values.astype('U')).toarray())
    h5f.close()

if not os.path.exists("data\\train_x2.h5"):
    print(2)
    cities = train_data["City"].values.astype('U')
    tfidf = CountVectorizer(max_features=50, binary=True, ngram_range=(1, 1))
    train_x2 = tfidf.fit_transform(cities)
    h5f = h5py.File("data\\train_x2.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x2.toarray())
    h5f.close()

    h5f = h5py.File("data\\test_x2.h5", 'w')
    test_x2 = tfidf.transform(test_data["City"].values.astype('U')).toarray()
    h5f.create_dataset('dataset_1', data=test_x2)
    h5f.close()

if not os.path.exists("data\\train_x3.h5"):
    print(3)
    name = train_data["NameOfCompany"].values.astype('U')
    tfidf = CountVectorizer(min_df=300 / len(name), binary=True, ngram_range=(1, 2))
    train_x3 = tfidf.fit_transform(name)
    print(3)
    h5f = h5py.File("data\\train_x3.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x3.toarray())
    h5f.close()
    h5f = h5py.File("data\\test_x3.h5", 'w')
    h5f.create_dataset('dataset_1', data=tfidf.transform(test_data["NameOfCompany"].values.astype('U')).toarray())
    h5f.close()

if not os.path.exists("data\\train_x4.h5"):
    train_x4 = train_data[["Exp", "EmploymentType", "WorkHours"]].as_matrix()
    h5f = h5py.File("data\\train_x4.h5", 'w')
    h5f.create_dataset('dataset_1', data=train_x4)
    h5f.close()

    test_x4 = test_data[["Exp", "EmploymentType", "WorkHours"]].as_matrix()
    h5f = h5py.File("data\\test_x4.h5", 'w')
    h5f.create_dataset('dataset_1', data=test_x4)
    h5f.close()

if not os.path.exists("data\\train_y.h5"):
    mx = max(max(y_test), max(y_train))
    mn = min(min(y_test), min(y_train))
    y_train = y_train / (mx - mn)

    h5f = h5py.File("data\\train_y.h5", 'w')
    h5f.create_dataset('dataset_1', data=y_train)
    h5f.close()

    y_test = y_test / (mx - mn)
    h5f = h5py.File("data\\test_y.h5", 'w')
    h5f.create_dataset('dataset_1', data=y_test)
    h5f.close()
