import sqlite3
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

data_path = os.path.split(os.getcwd())[0] + "\\data"
test_train_path = os.path.split(os.getcwd())[0] + "\\test_train\\"

version = ["", "_normalized"]
for i in version:
    cnx = sqlite3.connect(data_path + "\\vacancies3" + i + ".db")
    data = pd.read_sql_query("SELECT * FROM hh", cnx)
    cnx.close()
    data = data[data["TextOfVacancy"] != '']
    columns_y = ["up", "down"]
    columns_x = []
    for c in data.columns.values:
        if c not in columns_y+["NameOfCompany"]:
            columns_x.append(c)
    columns_x.append("mean_sal")
    for j in columns_y:
        new_data = data[data[j] != 0]
        unique_companies = list(set(list(new_data["NameOfCompany"])))
        companies = new_data["NameOfCompany"].as_matrix()
        x = new_data.drop(["up", "down", "NameOfCompany"], axis=1).as_matrix()
        y = new_data[j].as_matrix()

        train_size = int(0.8*x.shape[0])
        indexies = np.arange(x.shape[0])
        np.random.shuffle(indexies)
        train_indexies = np.sort(indexies[:train_size])
        test_indexies = np.sort(indexies[train_size:])
        x_train = x[train_indexies]
        y_train = y[train_indexies]
        x_test = x[test_indexies]
        y_test = y[test_indexies]
        train_companies = companies[train_indexies]
        test_companies = companies[test_indexies]
        mean_sal = {unique_companies[i]: [0, 0] for i in range(len(unique_companies))}

        for k in range(train_indexies.shape[0]):
            tmp = mean_sal[train_companies[k]]
            mean_sal[train_companies[k]] = [tmp[0]+y_train[k], tmp[1]+1]
        for k in unique_companies:
            v = mean_sal[k]
            if not v[1]:
                mean_sal[k] = v[0]
            else:
                mean_sal[k] = v[0] / v[1]

        data_sal = []
        for k in train_companies:
            data_sal.append(mean_sal[k])

        data_sal = np.array(data_sal).reshape((len(train_companies), 1))
        x_train = np.concatenate([x_train, data_sal], axis=1)

        data_sal = []
        for k in test_companies:
            data_sal.append(mean_sal[k])

        data_sal = np.array(data_sal).reshape((len(test_companies), 1))
        x_test = np.concatenate([x_test, data_sal], axis=1)

        pd.DataFrame(x_train, columns=columns_x).to_csv("{0}{1}_x_train{2}.csv".format(test_train_path, j, i),
                                                        encoding="utf-8", index=False)
        pd.DataFrame(x_test, columns=columns_x).to_csv("{0}{1}_x_test{2}.csv".format(test_train_path, j, i),
                                                       encoding="utf-8", index=False)
        pd.DataFrame(y_train, columns=[j]).to_csv("{0}{1}_y_train{2}.csv".format(test_train_path, j, i),
                                                  encoding="utf-8", index=False)
        pd.DataFrame(y_test, columns=[j]).to_csv("{0}{1}_y_test{2}.csv".format(test_train_path, j, i),
                                                 encoding="utf-8", index=False)
