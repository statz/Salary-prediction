import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sys
sys.path.append(os.getcwd())
from ga import GA

ud = ["down", "up"]
for l in ud:
    data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
    gl_train_data_x = pd.read_csv(data_path + l + "_x_train_normalized.csv")
    gl_train_y = pd.read_csv(data_path + l + "_y_train_normalized.csv")[l]
    df = pd.DataFrame()

    for cl in range(400):
        print("######################")
        print("Cluster #"+str(cl))
        print("######################")
        X = gl_train_data_x[gl_train_data_x["main"] == cl]["TextOfVacancy"].as_matrix()
        ind_train_x = []
        cl_train = gl_train_data_x["main"].tolist()
        for i in range(len(cl_train)):
            if cl_train[i] == cl:
                ind_train_x.append(i)
        Y = gl_train_y.iloc[ind_train_x].as_matrix()

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8,  random_state=0)

        words = {}
        for i in range(x_train.shape[0]):
            lex = str(x_train[i]).split(" ")
            for j in lex:
                if words.get(j) is None:
                    words.update({j: 1})
                else:
                    words.update({j: words.get(j) + 1})
        nw = []
        for w in words:
            k = words.get(w)
            if k > 10 and k < 2000:
                nw.append(w)

        words = nw
        nw = None

        binary_train_x = np.zeros([len(y_train), len(words)])
        binary_test_x = np.zeros([len(y_test), len(words)])

        d = {words[w]: w for w in range(len(words))}
        words = set(words)
        for i in range(len(y_train)):
            tokens = str(x_train[i]).split(" ")
            for j in tokens:
                if j in words:
                    binary_train_x[i, d.get(j)] = 1

        for i in range(len(y_test)):
            tokens = str(x_test[i]).split(" ")
            for j in tokens:
                if j in words:
                    binary_test_x[i, d.get(j)] = 1

        ga = GA(50, 10, len(words), 0.05, linear_model.SGDRegressor(l1_ratio= 1, penalty="none", n_iter=20))
        res = ga.optimize(binary_train_x, y_train, binary_test_x, y_test)
        ind = np.array(np.where(res))
        ind = ind.reshape(ind.shape[1])
        ndf = pd.DataFrame(np.array(list(words))[ind], columns=[str(cl)])
        df = pd.concat([df, ndf], axis=1)
    df.to_csv(os.getcwd()+"\\data\\"+l+"_words.csv", index=False, encoding="utf-8")