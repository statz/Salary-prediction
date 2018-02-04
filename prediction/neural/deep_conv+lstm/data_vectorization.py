from gensim.models import Word2Vec as w2v
import pandas as pd
import os
import numpy as np
import h5py

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
ud = ["down"]
for l in ud:
    train_data_x = pd.read_csv(data_path + l+"_x_train.csv")["TitleOfVacancy"].astype(str)
    test_data_x = pd.read_csv(data_path + l+"_x_test.csv")["TitleOfVacancy"].astype(str)
    cl = []
    for i in train_data_x:
        cl.append(len(str(i).split(" ")))
    median_len = 4
    ind_train_x = np.zeros([len(train_data_x), median_len])
    model = w2v.load(os.getcwd()+"\\data\\"+l+"_model")
    vocabulary = list(model.wv.vocab)
    d = {vocabulary[w]: w for w in range(len(vocabulary))}
    vocabulary = set(vocabulary)
    for i in range(train_data_x.shape[0]):
        print(i)
        words = str(train_data_x.iloc[i]).split(' ')
        j = 0
        for k in range(len(words)):
            if words[k] in vocabulary:
                 ind_train_x[i][j] = d[words[k]]
                 j += 1
                 if j == median_len:
                     break
    h5f = h5py.File(os.getcwd() + "\\data\\train_x1.h5", 'w')
    h5f.create_dataset('dataset_1', data=ind_train_x)
    h5f.close()
    print("1")
    ind_test_x = np.zeros([len(test_data_x), median_len])
    for i in range(test_data_x.shape[0]):
        print(i)
        words = str(test_data_x.iloc[i]).split(' ')
        j = 0
        for k in range(len(words)):
            if words[k] in vocabulary:
                ind_test_x[i][j] = d[words[k]]
                j += 1
                if j == median_len:
                    break
    h5f = h5py.File(os.getcwd() + "\\data\\test_x1.h5", 'w')
    h5f.create_dataset('dataset_1', data=ind_test_x)
    h5f.close()

    print("2")
    vectors_repr = np.empty([len(vocabulary), 50])
    i = 0
    for w in vocabulary:
        vectors_repr[i, :] = model.wv[w]
        i += 1
    h5f = h5py.File(os.getcwd()+"\\data\\" + l + "_vectors_repr.h5", 'w')
    h5f.create_dataset('dataset_1', data=vectors_repr)
    h5f.close()

