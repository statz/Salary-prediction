from gensim.models import Word2Vec as w2v
import pandas as pd
import os
import numpy as np
import h5py

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
ud = ["up", "down"]
for l in ud:
    train_data_x = pd.read_csv(data_path + l+"_x_train.csv")["TextOfVacancy"]
    test_data_x = pd.read_csv(data_path + l+"_x_test.csv")["TextOfVacancy"]
    #train_y = pd.read_csv(data_path + l + "_y_train.csv")[l].as_matrix()
    #test_y = pd.read_csv(data_path + l + "_y_test.csv")[l].as_matrix()

    median_len = int(np.median(list(map(len, train_data_x))))
    vect_train_x = np.zeros([len(train_data_x), 50*200])
    model = w2v.load(os.getcwd()+"\\data\\"+l+"_model")
    for i in range(train_data_x.shape[0]):
        words = str(train_data_x.iloc[i]).split(' ')
        j = 0
        for k in range(len(words)):
            if words[k] in model.wv.vocab:
                 vect_train_x[i][50*j:50*(j+1)] = model.wv[words[k]]
                 j += 1
                 if j == 200:
                     break
    print("2123")
    h5f = h5py.File(os.getcwd()+"\\data\\"+l+"_vect_train_x.h5", 'w')
    h5f.create_dataset('dataset_1', data=vect_train_x)
    h5f.close()
    print("0")
    del vect_train_x
    print("1")
    vect_test_x = np.zeros([len(test_data_x), 100*median_len])
    for i in range(test_data_x.shape[0]):
        words = str(test_data_x.iloc[i]).split(' ')
        j = 0
        for k in range(len(words)):
            if words[k] in model.wv.vocab:
                vect_test_x[i][50 * j:50 * (j + 1)] = model.wv[words[k]]
                j += 1
                if j == 200:
                    break
    h5f = h5py.File(os.getcwd()+"\\data\\"+l+"_vect_test_x.h5", 'w')
    h5f.create_dataset('dataset_1', data=vect_test_x)
    h5f.close()
    del vect_test_x
    print("2")